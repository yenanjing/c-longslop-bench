# Ingress 流量整形

- **作者**: laixintao(卡瓦邦噶)
- **年代**: authentic_human_blog
- **链接**: https://www.kawabangga.com/posts/7376

---

问题是这样，我们有 2 组服务器，A 组和 B 组，一部分用户从 A 组下载文件，一部分从 B 组下载文件。分成 2 组的初衷是，B 组用户下载量通常比较大，并且是离线服务，意味着可以慢一些，但不能影响 A 组用户。现在的问题是，A 组 和 B 组最终下载的来源都是 S3，依然存在共享资源，会出现问题：在下载量比较大的时候，占用全部的 S3 上传带宽，导致 A 组用户收到影响。

我们希望限制 B 到 S3 的带宽，比如 10Gbps.

有几个限制：

我们无法在 Client – Server 这里限流，因为 Server 侧有 Cache，我们希望 Client – Server 这里依然可以使用非常大的带宽来下载；

S3 的服务器没有限流的功能，这些服务器我们也无法控制，我们只能配置 Server A 和 Server B；

这个问题的难点在于：流量整形一般都是做在 egress 的，因为 egress 天然有一个 queue，发送端可以决定在什么时候发出去 packet，从而把发送速率可以稳定控制在一个想要的比例。

但是我们这里的场景是 ingress，对于 ingress 是没有 queue 的，流量整形一般也无法作用在 ingress ——因为包已经被物理网卡收到，DMA 完成，作为 ingress 侧是无法决定这个包什么时候到达的，只能决定怎么处理这个包。所以 ingress 侧一般是 classification, action，即 drop。

基于 Drop 的 policing（真形象的名字，超速开罚单的警察） 就有一个问题：由于 TCP 的拥塞控制机制，发送的带宽会逐渐上涨，最终会超过 10Gb。限流开始生效时，sender 瞬时发送速率超过 10 Gbit/s，并持续消耗 policer 的 token；当 token 不足时，后续 packet 被判定为 exceed 并丢弃。

大量包被丢弃导致 TCP 的发送端认为发生了拥塞，会大幅下降 cwnd。实际测试，如果用 policing，实际能跑的速度大概在 5-7Gb，而且不稳定。

Policing 和 Shaping 的区别和效果，Cisco 博客的这张图非常直观：

image source

以下是一个模拟环境，使用 policing 限速在 1Gbps （为了模拟环境里面用 wireshark 分析速度快一些，这里用 1Gbps 的限速为目标），其 throughput 如下图所示：

可以看到限速在 1Gbps 的时候，实际速度在 850Mbps 左右震荡

Window Scaling 图

再看 window scaling，可以看到 outstanding bytes 的上包络线呈现出明显的 AIMD/CUBIC loss-response 特征；结合 sender 侧的 ss -ti 可以确认 cwnd 在 loss 后也发生了类似下降——在上升到一定的大小之后，就会因为超过 policing 的限速而被 drop 一些，TCP 在发现有丢包之后，就降低 cwnd，从而降低发送速度，符合 AIMD。（这个环境用的拥塞算法是 cubic，默认的参数 cat /sys/module/tcp_cubic/parameters/beta 是 717，下降的比例是 717/1024 ≈ 0.7，和图中相符，图中是从 500kb 下降到 350kb）。

这样用户就不高兴了：为什么实际只能用到 5Gb 左右，不是说好了 10Gb 的吗？

如果要做到规整的 10Gb，这里就需要用到流量整形了，traffic shaping。

Linux 有一个特殊的驱动：ifb，全称是 Intermediate Functional Block，是一个虚拟网络设备，主要用来把原本的 ingress 流量重定向到一个“可做 egress qdisc”的设备上，从而实现 ingress shaping。

本质上，就是在收包链路上加一个网卡绕一下，这个网卡会调用 egress 的代码，这样我们就可以使用 tc 原来的逻辑做 egress 的 traffic shaping 了。

加上 ifb 虚拟网卡之后，包的路线如下：

IFB 加入之后的包路线

在 ifb_xmit() 的时候，看起来是一个负责发送包的代码，实际上它的逻辑是：修改 skb 的 dev 为原来真实的进来的那个 dev1，（而不是 ifb 这个虚拟的），然后进入到 netif_receive_skb() kernel 网络栈里面。

这样，我们就可以用这个 egress 的 qdisc 做 shaping 了。

实现的代码如下：

第一步，创建 ifb 虚拟接口，并且把流量从物理网卡 redirect 到 ifb：

第二步，创建一个 HTB，有 2 个 class，一个有限速 10Gb，另一个没有，可以跑到物理线速。

第三步，把要限速的 IP 放到 1:10 class：

然后测试吞吐，证明实际的 throughput 稳定在 956 Mbits/sec。

在 ifb 参与的 traffic shaping 下，限速可以稳定在 950 Mbits

cwnd 也在稳定地上涨

这样就达到了我们的目的：这条链路的速度稳定地使用限速 1Gbps 左右。

有几个小问题值得讨论。

为什么是 950 Mbits/sec 而不是 1Gbits/sec？

tc 做限制的时候，计算的是 skb 的 size，1500 Bytes，而这张图（以及使用的 iperf3 的显示）是计算的 TCP 的 body，即 MSS，1448 Bytes。1000 × 1448 / 1500 ≈ 965 Mbit/s。

在 cwnd 的图里面，刚开始的时候有一些丢包，为什么？

TCP 启动时通常先经过 slow start，cwnd 快速增长；进入 congestion avoidance 后，CUBIC 按三次函数继续探测可用带宽。发生 loss 后，它记录此前的窗口位置并降低 cwnd，随后重新逼近旧的 Wmax。

在这个图里面，只有一开始的时候有 loss，后续几乎没有 loss，贴着 throughput 发，说明 traffic shaping 做的不错。

按照 TCP 的拥塞控制算法，AIMD，即使是 traffic shaping 不也应该在最大窗口附近震荡吗？为什么这个图看起来这么平滑？

一个原因是，在逐渐逼近 1Gbps 的时候，limit 已经不是 cwnd 了，这种情况下 cubic 不会再激进地增加 cwnd2。通过 ss -ti 可以看到，大部分的时间卡在 sndbuf_limited:4688ms(29.6%)。

如果调整一下参数，扩大 wmem，cwnd 实际也会震荡的。

最后一列是 cwnd，也在不断震荡

最后一个问题也是最关键的问题，为什么在 traffic shaping 之后，cwnd 的变化速度变得这么平滑了呢？都是一样的 cubic 算法，为什么不像 policing 那样剧烈下降？以及为什么这里 send buffer 出现了瓶颈？

使用 policing 时，包进入 ingress 只有 2 中 action：

接受，然后 kernel 的网络栈会 ack，处理包；

直接 drop，没有 ack，导致 cwnd 下降。

使用 ifb 的 traffic shaping 之后，ingress 的包进入物理网卡，被 mirror 到 ifb，然后在这里经过 queue 排队。经过流量整形之后，包到达 kernel 的网络栈，然后发出 ack。这里的关键是，tc 添加了一个漏斗，大量的包（当然，buffer 需要内存的，内存是有限的，如果大到一定的程度就只能 drop 了）进来之后都存起来，然后用固定的流速放出去。只有被放出去的包才进入网络栈，只有进入了网络栈，kernel 才会给 sender 发回去 ack，此时，sender 的 cwnd 才有了新的空闲，新的包得以发送。

再换一个角度解释一下，从 sender 的角度，假设 cwnd 是 4，sender 发送了 4 个包，在收到 ACK 之前，无法再发送新的包了。这时候，receiver 的 shaper 每秒通过一个包，即每秒会发回来一个 ack，这样，后续的每一秒，比如5，6，7，8，sender 都可以再发送一个包到网络上。

send buffer 是另一层限制。TCP 已经发送但尚未被 ACK 的数据仍然需要保留在 send buffer 中，以便发生丢包时重传。因此 IFB shaping 不仅延迟 ACK、限制 cwnd 中空间的释放，也会延迟 TCP send buffer 中已发送数据的释放。当 send buffer 不够大时，sender 甚至可能在 cwnd 尚未用满之前就受到 send buffer 限制。

所以 IFB 以固定速率释放数据后，ACK 也以相应的节奏返回。ACK 一方面释放 cwnd 中的 outstanding 空间，另一方面使已经确认的数据可以从 send buffer 中清除。两者共同使 sender 的发送节奏逐渐跟随 shaper 的速率。

即，网络上最多可以飞多少，是 cwnd/rwnd 等窗口决定；什么时候可以继续注入新 packet，则由 ACK clock 驱动。（而 ACK 也会让总的 cwnd 提高）。

以上就叫做 TCP 的 self-clocking3，ACK 本身就像“时钟脉冲”，会决定发送速度。

这个机制有一个关键就是包没有被丢弃，只是 ack 的时间晚了一些，如果包被丢了（比如 policing），那么 cwnd 会下降，无法匀速发送。

︎

if (!tcp_is_cwnd_limited(sk)) return;  ︎

在 Computer Networking: A Top-Down Approach Chapter 3.7 也有介绍 ︎

相关文章:

由一个子网掩码配置错误所想到的

一次内存引起的网络丢包问题排查

网络中的环路和防环技术

服务器高性能网络调优

Traceroute the World
