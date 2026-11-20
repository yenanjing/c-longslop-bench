# Go 编程模式：k8s Visitor 模式

- **作者**: 陈皓(左耳朵耗子)
- **年代**: pre_ai
- **链接**: https://coolshell.cn (2020/Go%20%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%BC%8F%EF%BC%9Ak8s%20Visitor%20%E6%A8%A1%E5%BC%8F.md)

---

Go 编程模式：k8s Visitor 模式
>date: 2020-12-26T19:25:46+08:00

本篇文章主要想讨论一下，Kubernetes 的 `kubectl` 命令中的使用到到的一个编程模式 – Visitor（注：其实，`kubectl` 主要使用到了两个一个是Builder，另一个是Visitor）。本来，Visitor 是面向对象设计模英中一个很重要的设计模款（参看Wikipedia Visitor Pattern词条），这个模式是一种将算法与操作对象的结构分离的一种方法。这种分离的实际结果是能够在不修改结构的情况下向现有对象结构添加新操作，是遵循开放/封闭原则的一种方法。这篇文章我们重点看一下 `kubelet` 中是怎么使用函数式的方法来实现这个模式的。

本文是全系列中第9 / 10篇：Go编程模式

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

* 一个简单示例
* k8s相关背景
* kubectl的实现方法
+ Visitor模式定义
+ Name Visitor
+ Other Visitor
+ Log Visitor
+ 使用方代码
+ Visitor修饰器

一个简单示例

我们还是先来看一个简单设计模式的Visitor的示例。

* 我们的代码中有一个`Visitor`的函数定义，还有一个`Shape`接口，其需要使用 `Visitor`函数做为参数。
* 我们的实例的对象 `Circle`和 `Rectangle`实现了 `Shape` 的接口的 `accept()` 方法，这个方法就是等外面给我传递一个Visitor。

然后，我们实现两个Visitor，一个是用来做JSON序列化的，另一个是用来做XML序列化的

下面是我们的使用Visitor这个模式的代码

其实，这段代码的目的就是想解耦 数据结构和 算法，使用 Strategy 模式也是可以完成的，而且会比较干净。**但是在有些情况下，多个Visitor是来访问一个数据结构的不同部分，这种情况下，数据结构有点像一个数据库，而各个Visitor会成为一个个小应用。** `kubectl`就是这种情况。

k8s相关背景

接下来，我们再来了解一下相关的知识背景：

* 对于Kubernetes，其抽象了很多种的Resource，比如：Pod, ReplicaSet, ConfigMap, Volumes, Namespace, Roles …. 种类非常繁多，这些东西构成为了Kubernetes的数据模型（点击 Kubernetes Resources 地图 查看其有多复杂）
* `kubectl` 是Kubernetes中的一个客户端命令，操作人员用这个命令来操作Kubernetes。`kubectl` 会联系到 Kubernetes 的API Server，API Server会联系每个节点上的 `kubelet` ，从而达到控制每个结点。
* `kubectl` 主要的工作是处理用户提交的东西（包括，命令行参数，yaml文件等），然后其会把用户提交的这些东西组织成一个数据结构体，然后把其发送给 API Server。
* 相关的源代码在 `src/k8s.io/cli-runtime/pkg/resource/visitor.go` 中（源码链接）

`kubectl` 的代码比较复杂，不过，其本原理简单来说，**它从命令行和yaml文件中获取信息，通过Builder模式并把其转成一系列的资源，最后用 Visitor 模式模式来迭代处理这些Reources**。

下面我们来看看 `kubectl` 的实现，为了简化，我用一个小的示例来表明 ，而不是直接分析复杂的源码。

kubectl的实现方法

Visitor模式定义

首先，`kubectl` 主要是用来处理 `Info`结构体，下面是相关的定义：

我们可以看到，

* 有一个 `VisitorFunc` 的函数类型的定义
* 一个 `Visitor` 的接口，其中需要 `Visit(VisitorFunc) error`  的方法（这就像是我们上面那个例子的 `Shape` ）
* 最后，为`Info` 实现 `Visitor` 接口中的 `Visit()` 方法，实现就是直接调用传进来的方法（与前面的例子相仿）

我们再来定义几种不同类型的 Visitor。

Name Visitor

这个Visitor 主要是用来访问 `Info` 结构中的 `Name` 和 `NameSpace` 成员

我们可以看到，上面的代码：

* 声明了一个 `NameVisitor` 的结构体，这个结构体里有一个 `Visitor` 接口成员，这里意味着多态。
* 在实现 `Visit()` 方法时，其调用了自己结构体内的那个 `Visitor`的 `Visitor()` 方法，这其实是一种修饰器的模式，用另一个Visitor修饰了自己（关于修饰器模式，参看《Go编程模式：修饰器》）

Other Visitor

这个Visitor主要用来访问 `Info` 结构中的 `OtherThings` 成员

实现逻辑同上，我就不再重新讲了

Log Visitor

使用方代码

现在我们看看如果使用上面的代码：

上面的代码，我们可以看到

* Visitor们一层套一层
* 我用 `loadFile` 假装从文件中读如数据
* 最后一条 `v.Visit(loadfile)` 我们上面的代码就全部开始激活工作了。

上面的代码输出如下的信息，你可以看到代码的执行顺序是怎么执行起来了

我们可以看到，上面的代码有以下几种功效：

* 解耦了数据和程序。
* 使用了修饰器模式
* 还做出来pipeline的模式

所以，其实，我们是可以把上面的代码重构一下的。

Visitor修饰器

下面，我们用修饰器模式来重构一下上面的代码。

上面的代码并不复杂，

* 用一个 `DecoratedVisitor` 的结构来存放所有的`VistorFunc`函数
* `NewDecoratedVisitor` 可以把所有的 `VisitorFunc`转给它，构造 `DecoratedVisitor` 对象。
* `DecoratedVisitor`实现了 `Visit()` 方法，里面就是来做一个for-loop，顺着调用所有的 `VisitorFunc`

于是，我们的代码就可以这样运作了：

是不是比之前的那个简单？注意，这个`DecoratedVisitor` 同样可以成为一个Visitor来使用。

好，上面的这些代码全部存在于 `kubectl` 的代码中，你看懂了这里面的代码逻辑，相信你也能够看懂 `kubectl` 的代码了。
