# Go编程模式：Pipeline

- **作者**: 陈皓(左耳朵耗子)
- **年代**: pre_ai
- **链接**: https://coolshell.cn (2020/Go%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%BC%8F%EF%BC%9APipeline.md)

---

Go编程模式：Pipeline
>date: 2020-12-26T17:04:59+08:00

本篇文章，我们着重介绍Go编程中的Pipeline模式。对于Pipeline用过Unix/Linux命令行的人都不会陌生，他是一种把各种命令拼接起来完成一个更强功能的技术方法。在今天，流式处理，函数式编程，以及应用网关对微服务进行简单的API编排，其实都是受pipeline这种技术方式的影响，Pipeline这种技术在可以很容易的把代码按单一职责的原则拆分成多个高内聚低耦合的小模块，然后可以很方便地拼装起来去完成比较复杂的功能。

本文是全系列中第8 / 10篇：Go编程模式

* Go编程模式：切片，接口，时间和性能
* Go 编程模式：错误处理
* Go 编程模式：Functional Options
* Go编程模式：委托和反转控制
* Go编程模式：Map-Reduce
* Go 编程模式：Go Generation
* Go编程模式：修饰器
* Go编程模式：Pipeline
* Go 编程模式：k8s Visitor 模式
* Go编程模式 ： 泛型编程

« 上一篇文章下一篇文章 »

目录

* HTTP 处理
* Channel 管理
+ Channel转发函数
+ 平方函数
+ 过滤奇数函数
+ 求和函数
* Fan in/Out
* 延伸阅读

HTTP 处理

这种Pipeline的模式，我们在《Go编程模式：修饰器》中有过一个示例，我们在这里再重温一下。在那篇文章中，我们有一堆如 `WithServerHead()` 、`WithBasicAuth()` 、`WithDebugLog()`这样的小功能代码，在我们需要实现某个HTTP API 的时候，我们就可以很容易的组织起来。

原来的代码是下面这个样子：

通过一个代理函数：

我们就可以移除不断的嵌套像下面这样使用了：

Channel 管理

当然，如果你要写出一个泛型的pipeline框架并不容易，而使用Go Generation，但是，我们别忘了Go语言最具特色的 Go Routine 和 Channel 这两个神器完全也可以被我们用来构造这种编程。

Rob Pike在 Go Concurrency Patterns: Pipelines and cancellation 这篇blog中介绍了如下的一种编程模式。

Channel转发函数

首先，我们需一个 `echo()`函数，其会把一个整型数组放到一个Channel中，并返回这个Channel

然后，我们依照这个模式，我们可以写下这个函数。

平方函数

过滤奇数函数

求和函数

然后，我们的用户端的代码如下所示：（注：**你可能会觉得，`sum()`，`odd()` 和 `sq()`太过于相似。你其实可以通过我们之前的Map/Reduce编程模式或是Go Generation的方式来合并一下**）

上面的代码类似于我们执行了Unix/Linux命令： `echo $nums | sq | sum`

同样，如果你不想有那么多的函数嵌套，你可以使用一个代理函数来完成。

然后，就可以这样做了：

Fan in/Out

动用Go语言的 Go Routine和 Channel还有一个好处，就是可以写出1对多，或多对1的pipeline，也就是Fan In/ Fan Out。下面，我们来看一个Fan in的示例：

我们想通过并发的方式来对一个很长的数组中的质数进行求和运算，我们想先把数组分段求和，然后再把其集中起来。

下面是我们的主函数：

再看我们的 `prime()` 函数的实现 ：

我们可以看到，

* 我们先制造了从1到10000的一个数组，
* 然后，把这堆数组全部 `echo`到一个channel里 – `in`
* 此时，生成 5 个 Channel，然后都调用 `sum(prime(in))` ，于是每个Sum的Go Routine都会开始计算和
* 最后再把所有的结果再求和拼起来，得到最终的结果。

其中的merge代码如下：

用图片表示一下，整个程序的结构如下所示：

延伸阅读

如果你还想了解更多的这样的与并发相关的技术，可以参看下面这些资源：

* **Go** **Concurrency** **Patterns** – **Rob** **Pike –** 2012 Google I/O presents the basics of Go‘s concurrency primitives and several ways to apply them.

* **Advanced Go Concurrency Patterns**– **Rob** **Pike** – 2013 Google I/O covers more complex uses of Go’s primitives, especially select.

* **Squinting at Power Series**– **Douglas McIlroy**‘s paper shows how Go-like concurrency provides elegant support for complex calculations.
