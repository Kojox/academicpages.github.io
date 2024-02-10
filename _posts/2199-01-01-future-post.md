---
title: 'Fast fluid simulation using convolutional neural networks'
date: 2024-02-10
permalink: /posts/2012/08/blog-post-4/
tags:
  - cool posts
  - SPH
  - simulation
  - deep learning
  - fluid simulation
---

Hello! The great thing about doing gamedev is that almost every aspect of computer science is applicable in some way or form, so there is always something cool and completely new to learn 😊. And it is no different here. Although I can imagine that you might be wondering why game development is even mentioned in a blogpost about a deep learning technique. Well, that’s because deep learning finds almost a way into every creek of game development. From Nvidia’s DLSS, to Forza’s Drivatars, or even LLM powered NPCs interacting with the player.


If you want to get a better understanding and see many applications of machine learning in games, I can only recommend this YouTube channel called [AIandGames](https://www.youtube.com/@AIandGames/videos) to you. In his video [How Machine Learning is Transforming the Video Games Industry](https://www.youtube.com/watch?v=dm_yY-hddvE) you can get a great overview about how deep learning is used across the board.

In this blog post I would like to take you into a dive into fluid simulations. Why fluid simulations you ask? Ah let me show you something that has been bothering me for a while.

First let me start of by showing you how important good physics are to games (yes even more then graphics). In the end games are all about interactions with the player. If the player performs an action in the world, let’s say shoots a rocket launcher at a wall and then the game world does not react to it, the player feels fooled or the immersion gets broken. Now this is not only putting all the weight onto the physics. Graphics, audio and even haptics etc. do need to support this interaction too (the sound of the explosion, a fire ball lighting up the scene, a decal marking the impact, ...) But physics are the backbone of every game-object bouncing around which the player interacts with.



<!-- 
<div style="text-align:center">
            <img src="/images/teardown-pipe.gif" alt="scene_graphic" style="width:96%;height:96%">
    <br>
</div>
-->


<div style="text-align:center">
        <img src="/images/teardown-pipe.gif" alt="scene_graphic" style="width:48%;height:48%">
        <img src="/images/teardown-foam.gif" alt="scene_graphic" style="width:48%;height:48%">
</div>
&nbsp;

A great example of a game, which already does some very nice physics and puts the interaction of the player with the physics first, is Teardown. Don’t let the voxel style fool you. When it comes to rigid body simulation and destruction there are some beautiful techniques used. If you are interested into that there is a dev talk which goes a bit into the simulation aspect of the game. [Teardown Engine Technical Dive]( https://www.youtube.com/watch?v=tZP7vQKqrl8)

Above you can see that not only do they simulate bending pipes, fire, smoke emitted from the fire, or the foam of a fire extinguisher.
But I obviously want to point out the one thing they do not simulate &rarr; __the water__

<div style="text-align:center">
            <img src="/images/teardown-box.gif" alt="scene_graphic" style="width:97%;height:97%">
    <br>
</div>
&nbsp;

Sure, buoyancy is there. But that’s it. No water being displaced by me swishing around in the water only graphics effects covering up that noting is being simulated. And I think that is such a shame, since the is so much potential lost. For example, you can’t simply go and splash a body of water onto a burning building to extinguish it.

But why is that? Well in games there is additionally a real-time constraint. Meaning that games, as already mentioned, are interactive in nature. To put things into perspective: a single frame of a game at 30 FPS allows the application a mere 33.3 milliseconds to process the entire update. This includes everything: physics, rendering, audio, input, ai and more. Dip below that and you got yourself nauseated and angry players. Usually, people even expect the game to run even faster than that. Of course, you can still employ tricks like updating the physics across multiple frames, but that can only go so far. If the water updates every 2 minutes you can just forget it. And that’s exactly what game developers have been doing. Either fluids are approximated or maybe just 2D simulations instead like in the game [Worms](https://www.youtube.com/watch?v=3SzFZg4lzRE).

Ok so now how could we solve this? Maybe we can use deep learning to accelerate the simulation code and allow for 3D fluid simulation in games.
I want to introduce you to a paper which is using convolutional neural networks (CNNs) to do exactly that. Now if this demo scene doesn’t motivate you in the possibilities this could offer...

<div style="text-align:center">
            <img src="/images/canyon.gif" alt="scene_graphic" style="width:100%;height:80%">
    <br>
    <!---<i>Figure 4: The authors goal for the architecture their network.</i>-->
</div>

This scene contains over 60.000 fluid particles which interact with not only each other but also with over 230.000 static particles. On average it took 180ms to calculate a new frame / timestep, which is also the framerate this animation is played at. All collision handling is performed by the underlying network architecture.

It has been shown that CNNs can achieve a speedup of a factor of two to four times compared to classical numerical flow solvers. Now I need to say that in the paper directly they never compare the times to classical solvers but rather to other deep learning approaches. But I did some digging and found a [comment](https://openreview.net/forum?id=B1lDoJSYDH) of the author stating that for a certain 16-second-long scene their solution ran in real-time, while the classical solver needed about 9 minutes to simulate the entire sequence. This sounds very promising to me.

The paper “Lagrangian Fluid Simulation with Continuous Convolutions” proposes a new way to do fluid simulations. Before we dive into the technicalities, lets take a quick look at the key promises of the paper. And see what else this paper can provide us with.
- A very interesting thing about this paper is that it learns from the data provided. This means it does not depend on some fancy and complicated physics modelling in the background. Rather it will learn to reproduce seen simulations.
- It can simulate fluids with different viscosity parameters. So, we can not only simulate Water but also more honey like fluids.
- Additionally, it generalizes to new and unseen scenarios very well. This is very nice because it doesn’t constrain us to work in a certain size of levels or shapes. Even allows for the geometry to be editable.
- It has long term stability. This is important. This guarantees us that the water will come to rest after a while and not become unstable after a certain time has passed. Otherwise maybe after 5 min all particles just go flying everywhere. Not only does this contradict the realistic behaviour one would expect from water, but it also means you couldn’t rely on it for important game design aspects. (Think of a puzzle with water and the water just disappears)

First, we will start with the basics of these simulations and then we will take a look at how deep learning can help us to speed up the simulations.

## Basics

The method I want to introduce to you in this blog post is an *meshless and particle-based method*. Let's get an overview of what this even means.

### Computational meshes
There are three grid types in which methods can be categorized into. These types can be seen in the following figure:

<div style="text-align:center">
        <img src="/images/structured.jpg" alt="scene_graphic" style="width:32%;height:32%">
        <img src="/images/unstructured.png" alt="scene_graphic" style="width:32%;height:32%">
        <img src="/images/meshless.png" alt="scene_graphic" style="width:32%;height:32%">
        <i>
        Figure 1: structured mesh (left), unstructured mesh (center), meshless(right). Source: 
        <a href="https://www.simscale.com/docs/simwiki/preprocessing/what-is-a-mesh/"> [1]</a>
        <a href="https://www.researchgate.net/figure/Unstructured-mesh-for-NACA-0012-cropped_fig3_228825923"> [2]</a>
        <a href="https://ericpko.github.io/projects/sph-sim/"> [3]</a>
        </i>
</div>

- *Structured* meshes are quite common since they are easy to manage. An example we all know would be the cartesian grid. The disadvantage with this type of grid is that the grid resolution is uniformly the same. Given a flow it happens often that it is mostly uniform where one could use a coarser resolution while at certain areas so much is happening, that one would need a very fine grid to represent it. A uniform grid would then lead to a lot of wasted computations in homogeneous areas. The other two mesh types are more adaptive and are sometimes preferred because of this.
- *Unstructured* meshes are more flexible but more complex to handle since now neighbourhoods are not given implicitly from structure as in the grid and might need to be calculated first.
- *Meshless* methods do as the name says and do not use a mesh at all. Instead, they define points in space at which information gets sampled at. Their neighbourhood is defined through a distance metric e.g. the Euclidian distance.


### Eulerian vs Lagrangian Fluid Simulation

When it comes to fluid simulation there are two main approaches one will come across.
The Eulerian approach can be seen as the approach when thinking about the fluid on a fixed grid or any fixed points in space. The grids positions never change and each timestep we solve the PDE at the same position for the evolving time.
 
Contrary, in the Lagrangian approach we have a set of individual fluid particles instead of fixed points or cells. Each particle in the set carries its properties like velocity with them and the position will change over time.
A good analogy for this is to imagine a crowded area. If you specifically follow a single person around and track his movement, velocity etc. it is the Lagrangian view. If you consider a specific point of the floor and track the velocity of the nearest person to that point, it is the Eulerian view.
In theory both approaches are equivalent to each other and additionally one can even choose to do a mix between both to increase the performance. Of course, each approach comes with its own benefits when it comes to computational efficiency.


### Smoothed Particle Hydrodynamics (SPH)
The traditional way to do fluid simulations is by a method called SPH. While it is not even necessary to really know how this method works, it helps to understand certain decision made in the paper later on, since it draws similarity to the classical approach. Instead of writing at length about a method that in the end is only used to generate the data for the network, I would like to refer you to a few very good resources.
- If you have some time at hand and want to understand SPH not only in a practical way, but also develop an intuition why and how it works. Then I can't recommend [the video of Sebastian Lague](https://www.youtube.com/watch?v=rSKMYc1CQHE) enough.
- If you are more into reading and also want to learn about how this is done with deep learning (in this case graph based networks) then [this blog series](https://inductiva.ai/blog/article/sph-1-a-smooth-motivation) is quite nice.
- If you already know SPH then there is also a [paper on Divergence-Free Smoothed Particle Hydrodynamics (DFSPH) here](https://animation.rwth-aachen.de/media/papers/2015-SCA-DFSPH.pdf). This method is used to generate the simulation data from which the neural network is later trained on.

<div style="text-align:center">
            <img src="/images/lid_driven.gif" alt="scene_graphic" style="width:97%;height:97%">
    <br>
    <i>Animation of the <a href="https://www.openfoam.com/documentation/tutorial-guide/2-incompressible-flow/2.1-lid-driven-cavity-flow"> lid-driven cavity</a> scenario. A simulation using the SPH method I did a while back. The colour represents the magnitude of the velocity. From blue toward red the velocity increases.</i>
</div>


#### BOUNDARIES
There are different ways how the boundaries are handled by SPH methods. And handling different boundary conditions can be pretty tricky with SPH simulations. Especially if you want to model [different boundary conditions like no-slip, or in/out-flow](https://en.wikipedia.org/wiki/Boundary_conditions_in_fluid_dynamics). Apart from implicitly enforcing boundaries by calculating if a particle is out of bounds and pushing it back in, another popular way is to place static particles at the boundaries. This method was also used in the animation above. As you can see on the edges the particles are fixed regarding their position and velocity. This way of handling the boundary is also used in the deep learning method described in this blog.


## Using Deep Learning
As you may have picked up already from the previous section there are two approaches how currently neural networks are designed to simulate fluids. The other approach is achieved by using graph neural networks (GNNs). These GNNs share properties with unstructured meshes and in terms of accuracy perform very well. A big advantage which is shared with all deep learning approaches is that they learn from the data given to them. This way they can simulate far more complex physics without an explicit written up model, which in some cases might not even to so easy to do.

The GNN approach comes with its own disadvantages, which mainly come from additional complexity when implementing them. The arbitrary and often changing edges lead to message-passing approaches between the particles. Additionally, the graph's edges need to be updated continuously as the particles move.

This method I will cover in this blog post does not use a graph at all and instead uses convolutional neural networks, which generally also makes this method faster and more suited for using them in interactive fashion, then the ones based on GNNs.

### How does it work?
First I want to give you a high level overview of the approach presented in the paper.
And I will do this by showing you the steps the simulator needs to do to compute a single time step, given we already have the deep learning method already available as a building block.
There are three main steps involved in the algorithm:

__Step 1__ Compute intermediate positions and velocities at beginning of each time step

$v^{n*}_i = v^n_i + \Delta t \ a_e$

$x^{n*}_i = x^n_i + \Delta t \ \frac{v_i^n + v_i^{n *}}{2}$

This is done following [Heun's method](https://en.wikipedia.org/wiki/Heun%27s_method). $a_e$ is an acceleration through which all external forces like gravitation are applied.

__Step 2__ Use the convolutional network to implement interactions between particles

$[\Delta x_1, ..., \Delta x_N] = ConvNet({p^n_1, ..., p^n_N}, {s_1, ..., s_M})$

The static particles $s_j = (x_j,[n_j])$ do not only consist of the positional information but also include the normals of the scene (vectors pointing perpendicular to the wall).

$p^n_i$ are the dynamic particles and are defined as $(x_i^n,[1,v_i^n, \nu_i])$. $x_i^n$ is the position of the particle $i$ at time step $n$. Additionally, the velocity $v_i^n$ and viscosity $\nu_i$ is included. Lastly, there is also the constant 1 present. Why this is, is explained in an answer [here](https://openreview.net/forum?id=B1lDoJSYDH). It's to avoid that a particle with a zero feature vector has no influence on the convolution result.

__Step 3__ Apply the correction to update positions and velocities for the next timestep

$x_i^{n+1} = x_i^{n *} + \Delta x_i$

$v_i^{n+1} = \frac{x_i^{n + 1} - x_i^n}{\Delta t}$

This is simply applying the correction term of the previous step. The correction of the position simply gets added onto the last position. The velocity is calculated as illustrated in Figure 2 by the green vector.

By repeatedly running these steps over and over again you can now advance the simulation step by step.


<div style="text-align:center">
            <img src="/images/steps.png" alt="scene_graphic" style="width:50%;height:40%">
    <br>
    <i>Figure 2: Overview of the steps involved for one particle. In the first (blue) Step the particle is moved independent of other particles or the scene geometry. In the second step (red) the network will determine the offset position, now considering all interactions with other fluid particles and the static particles representing the scene geometry. In the last step (green) this correction of step 2 is finally applied.</i>
</div>

#### Network architecture
First let us take a closer look at the inputs and outputs this network has.

The _output_ features were already mentioned in the previous section. These are position updates that are applied to the particle for the next timestep.

As _inputs_ there are two types of particle sets: fluid and static particles. As you can see in Figure 3 these sets are separately processed. Another thing worth noticing is that for the convolutions within the fluid particles the particle for which the offset is calculated is excluded. Instead, it is included as its own input through fully connected layers.
How the input feature vector looks like was also described in the previous section.

<div style="text-align:center">
            <img src="/images/network.png" alt="scene_graphic" style="width:100%;height:80%">
    <br>
    <i>Figure 3: Network architecture.</i>
</div>

The network consists of just 4 layers and 5 convolutions in between. While the paper is quite sporadic when it comes to explaining the reasons why it is the way it is, I will try to summarize all the important aspects that have been mentioned by the authors.
At the first level convolutions are calculated at dynamic particles location with the static and dynamic particles. The features of each single particle are fully connected. Then on the following levels convolutions are only performed on the dynamic particles. There is the residual connection from the second to third layer. On most of the additions there is this star which just means that the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function is used here.



### CONTINUOUS CONVOLUTIONS
Finally let’s take a look at the actual main contribution of this paper: *continuous convolutions*. As you have just seen they do have some big advantages if you consider the quite lean architecture of the network, combined with its ability to learn fluid simulations.

These advantages come from a few properties these convolutions have, if you are unsure what they mean, then I have linked various blog posts explaining them further:
- [spatial inductive bias](https://iclr-blogposts.github.io/2023/blog/2023/how-does-the-inductive-bias-influence-the-generalization-capability-of-neural-networks/)
- [permutation invariance](https://fabianfuchsml.github.io/permutationinvariance/)
- [translation equivariance](https://fabianfuchsml.github.io/equivariance1of2/)

The continuous convolutions are an extension of the classical convolution operator also used in various other fields in computer science.
Just as a quick reminder let’s go through the concept of how convolutions work and what they actually do.

#### CONVOLUTIONS

Following figure 4 you can see that convolutions take a patch of for example an image and then they perform a per element multiplication inside this patch with the kernel (also sometimes called filter in this ML context). So in this example the 7 will be multiplied with the -1 in the upper left corner of the patch and kernel. Same goes for the 5 and 0 in the central entry. Then all of these entries are summed up together, resulting in a single value.

<div style="text-align:center">
            <img src="/images/convolutions.png" alt="scene_graphic" style="width:100%;height:80%">
    <br>
    <i>Figure 4: Illustration of the discrete convolution opertation. <a href="https://cg.ivd.kit.edu/downloads/assignment3_GPUC.pdf">[Source]</a></i>
</div>


Depending on the kernel this value can mean different things. In Figure 5 we can see examples for different filters and how they can represent different attributes of the source image.
If we imagine our kernel aligning with the "edge detect” kernel, the resulting value will tell you how much of an edge at this patch’s location is. The larger the value is the more of an edge or the more the certain feature this filter represents is present in this path.
I think it is important to note that this way the convolution operator dose not only indicate if a feature is present but also where it occurs, since it operates on these patches where all values are spatially close.
Now in our network, these filters are not handcrafted but rather learned and if these filters are learned by our neural network, they can learn to detect arbitrary features that help to solve a given problem.

<div style="text-align:center">
            <img src="/images/filters.png" alt="scene_graphic" style="width:100%;height:80%">
    <br>
    <i>Figure 5: Different effects of various handcrafted kernels applied to one image. <a href="https://www.youtube.com/watch?v=NmLK_WQBxB4
    ">[Source]</a></i>
</div>

#### THE FILTER FUNCTION
As we just have seen the most important part of convolutions is the filter function that we want to learn. In the paper they list and apply these three properties, which the filter function should have:
- **continuous:** Firstly the filter needs to be made continuous, to sample the filter at arbitrary positions, linear interpolation is used when looking up filter values. 
- **spherical:** Similar to how the kernel in SPH is spherical, here we transform the regular grid shape into a circle. This is done by a ball to cube mapping, which allows to retain the regular grid structure for storing and interpolating filter values. Another neat effect of this mapping is that spherical coordinates are avoided. These are problematic due to singularities and would require special treatment, which this way is completely avoided.
- **continuous output:** The output of the convolution should be continuous as well. For that they use a radial window function with limited support. This smoothly fades out contribution of particles with increasing distance to the filter centre. This ensures output function stays continuous even if particles move in or out of the filter’s radius.

These properties will allow to apply the filter onto unstructured point clouds, which is basically what our fluid particles are. In figure 6 the application of these properties is visualized.


<div style="text-align:center">
            <img src="/images/c_filter.png" alt="scene_graphic" style="width:100%;height:80%">
    <br>
    <i>Figure 6: Properties applied to the filter. <a href="https://papertalk.org/papertalks/4152
    ">[modifed from]</a></i>
</div>

<!-- 
<div style="text-align:center">
            <img src="/images/pointcloud.png" alt="scene_graphic" style="width:50%;height:50%">
    <br>
    <i>Figure 7: Properties applied to the filter.</i>
</div>
-->

#### FROM THE DISCRETE CONVOLUTION OPERATOR TO CCNs
Let's start from the known discrete convolution operator and build up the CCN by modifying it step by step.

So the following equation is the discrete convolution operator that we have seen previously. $f$ is the input and $g$ is the filter function. $\tau$ is the shift vector, which in this case would be a pixel wise offset covering all offsets included in the patch.

$(f * g)(x) = \sum_{\tau \in \Omega}^{} f(x + \tau) g(\tau)$

Now as the first step we transform this idea to particles or point clouds.

$(f * g)(x) = \sum_{i \in N(x,R)}^{} f(x_i + \tau) g(x_i - x)$

Instead of evaluating the convolution at every discrete pixel position, the convolution is now evaluated on the set $N(x,R)$. This set is the radial neighbourhood around the point $x$ with the radius $R$. Other than that, not much has changed and I hope you see the similarity between these two terms only now we operate not on fixed grid positions but on a set of points which as a side note regarding to their positions are now continuous.

Next a new symbol is introduced:

$(f * g)(x) = \sum_{i \in N(x,R)}^{} f(x_i + \tau) g(\Lambda(x_i - x))$

The $\Lambda$ is a function an performs the previously mentioned _ball to cube_ mapping. In the following figure the transformation this function performs is illustrated. It represents a mapping from the unit ball to the unit cube. This allows to use the grid representation for storing and interpolating filter values.

<div style="text-align:center">
            <img src="/images/mapping.png" alt="scene_graphic" style="width:50%;height:50%">
    <br>
</div>

At this point, the convolution operation is still missing one of the previously listed properties and that’s the radial weighing function to ensure that the kernel has a has a smooth falloff.
Which again leads to the output staying continuous even if particles enter or leave the radial window. And the learned influence of a particle smoothly drops to zero with increased distance from the filter centre.

$(f * g)(x) = \frac{1}{a_n} \sum_{i \in N(x,R)}^{} a(x_i,x) f(x_i + \tau) g(\Lambda(x_i - x))$

With $a(x_i,x)$ we add a radial weighting function. There is some flexibility how this can be chosen. Like how in SPH the kernel too can be designed differently. In the most familiar shape, you can imagine this to be a gaussian shape.

<!--- //gaussian shape good, nice fall off but derivatoin at the center zero... -->

In the paper they choose $a(x_i,x)$ to be the following function:

$$
\begin{align*}
a(x_i,x) = \begin{cases}
    (1 - \frac{||x_i - x||^2_2}{R^2})^3 & \text{if } ||x_i - x||_2 < R \\
    0 & \text{otherwise.}
\end{cases}
\end{align*}
$$

which results in this behaviour:

<div style="text-align:center">
            <img src="/images/window_func.png" alt="scene_graphic" style="width:50%;height:50%">
    <br>
    <i>$a(x_i,x)$ with $R = 2$ </i>
</div>

Lastly $a_n$ is a normalization factor. It can be defined differently. For example one could use the sum of all $a(x_i, x)$ in a certain neighbourhood.
In the paper they just set the normalization to a fixed $a(x_i, x) = 1$. This leads to changes in the density remaining preserved as a feature aiding in the process of learning fluid dynamics, since the density of a fluid has importance how the next timestep will look like.

#### SUMMARY
Good! That’s it. Let us quickly recap what we have just built up in this chapter:
<div style="text-align:center">
            <img src="/images/overview.png" alt="scene_graphic" style="width:70%;height:70%">
    <br>
</div>

Given the discrete convolution we transformed it into something that can handle point clouds by iterating over all the points in a certain neighbourhood. We made it possible to sample the filter and any location by applying linear filtering. Then the ball to cube mapping is used to squish it into a spherical shape while retaining all the advantages of a grid representation. At last, the window function to introduce a smooth falloff with increasing distance.

### Experiments
Now that we understand the formulas and theory behind this method, let's take a look at how the author's implementation performed.

#### TRAINING
The training was done in a supervised fashion. They generated data using the DFSPH method. This data consists of a box shaped environment with randomized falling shapes inside of it. Example how such a scene looked like can be seen in the following figure. Of these scenes 2000 were generated for training and 300 more for testing.

During training they predicted the next two future timesteps given an configuration and used the combined loss defined as: $\mathcal{L} = \mathcal{L}^{n+1} + \mathcal{L}^{n+2}$
This loss was optimized over $50.000$ timesteps using a learning rate decay, starting from $0.001$ down to $1.56e-5$

Another very great thing for someone without a supercomputer in the backyard is that this model has been trained on a single GPU (RTX 2080Ti) in about a day.

<div style="text-align:center">
            <img src="/images/dataset.png" alt="scene_graphic" style="width:70%;height:70%">
    <br>
    <i>Falling shapes used as training data.</i>
</div>

#### LOSS FUNCTION
The loss function is defined as:

$\mathcal{L} = \sum_{i=1}^{N} \phi_i \Vert x^{n+1}_i - \hat{x}^{n+1}_i \Vert^\gamma_2$

This is in essence nothing more then the distance of the particle $x^{n+1}_i$ to the ground truth $\hat{x}^{n+1}_i$ after a single timestep.

$\phi_i$ is an individual weighting factor for each point and is defined as $\phi_i = exp(-\frac{1}{c} \| \mathcal{N}(x^{n *}_i) \|)$ where $c = 40$ is set to the average number of neighbours of a particle across their experiments. $\mathcal{N}(x^{n *}_i)$ is the current number of neighbouring particles for the particle $i$. This emphasizes loss for particles with fewer neighbours. Since particles near the water surface or near the scene geometry will have fewer neighbours, these will get emphasized more. This makes sense because exactly these particles are more important compared to fluid particles hidden somewhere in the centre of the fluid.

$\gamma$ is set to $0.5$ to make the loss function more sensitive to small motions. This is supposed to make the simulation more accurate for small fluid flows.

#### EVALUATION
The reason why this method is great is given in the following table, where the authors compared their method to other deep learning based methods.
<div style="text-align:center">
            <img src="/images/eval_table.png" alt="scene_graphic" style="width:90%;height:90%">
    <br>
</div>
This data was generated with the dam break scenario, in which in a cubical environment a smaller cube of water is released.
As you can see the main and most interesting takeaway is that the inference time is very low compared to other methods. Other than that, the accuracy over two future predicted frames is reported. This was done by using every fifth frame of the ground truth simulation and then comparing the deviation of the particles. From these numbers we can see that the method is not worse in terms of accuracy compared to these other methods.

<div style="text-align:center">
            <img src="/images/eval_acc.png" alt="scene_graphic" style="width:90%;height:90%">
    <br>
</div>

Further they compared their solution with respect to accuracy to the classical non deep learning based [DFSPH](https://animation.rwth-aachen.de/media/papers/2015-SCA-DFSPH.pdf) method. To do this they first ran DFSPH with a $\Delta t$ of 1ms and used this as ground truth. Then again ran it again with a $\Delta t$ of 5ms. This is the orange line in the graph. Now they ran their method with a timestep of $\Delta t = 20ms$.
As you can see their method does not perform perfectly here. Especially in the beginning where the fluid begins to collide with the scene geometry the error is higher. The good thing is that after a while when the fluid becomes calmer the error becomes more similar. Why especially these $\Delta t$ were chosen is not explained by the authors, except that the $20ms$ are the rate at which the samples for the training data were generated at. 

While all training data was generated from scenarios inside this box environment, regarding __generalization__ the authors have thrown their method into completely different scenarios as it was the case in the animation from the very beginning of this post. Furthermore, do these other scenarios contain a vastly different amount of particles.

### Conclusion
As everything in life, not everything is perfect. In conclusion what are the pros and cons to using this approach?

#### PROS
- First of all, this method performs very well even though its quite simple. As we have just seen the accuracy is good and the inference time is also still the fastest. When researching other methods I still have not found a method with a faster inference time, so it still outperforms graph-based frameworks when it comes to speed.
- Also, in scenarios where, new particles are emitted during simulation. Are here better handled since this can be a costly operation for methods that build and maintain an explicit graph structure.
- The method maintains good long-term stability.
- It generalizes very well.

#### CONS
But not everything is perfect with this method.
- First it is important to note that for the implementation still an efficient neighbourhood search, as with SPH too, is needed. Since this is in terms of big $\mathcal{O}$ complexity still the slowest part of SPH, you can not expect to gain massive performance compared to the simpler methods.
- It struggles to simulate fluids which are compressible. As this is not necessarily needed, we can look over this weakness.
- On the other hand, something quite substantial is how it handles ridged and deformable solids. This is from a [paper](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf) which uses a graph based neural network, which directly compared their method to the method here described:


<div style="text-align:center">
            <img src="/images/boxbath.gif" alt="scene_graphic" style="width:70%;height:60%">
    <br>
    <!---<i>Figure 4: The authors goal for the architecture their network.</i>-->
</div>

As you can clearly see, this does not look too good. So, if you plan to use this method it is a good idea to stick to water simulations and handle rigid bodies in another way.

### Outlook
Ok so although this post will take an end here there is still some interesting stuff to look forward to. Now that we have covered the theory it will be time to turn it into practice and (hopefully) benefit from the newfound performance.
And in a good old gamedev fashion we will do that in a future blog post from scratch using Vulkan compute shaders.
Once the basics are up and running there are a few more ways to go from there. For one there is still another paper from the same authors called [Guaranteed Conservation of Momentum for Learning Particle-based Fluid Dynamics](https://arxiv.org/abs/2210.06036) which is worth looking into and is kind of fixing the problems you have just seen in the conclusion above. Although this method is a bit slower than the one presented here, the method is similarly accurate to the GNS paper while still being faster.

Then implementing interaction with rigid bodies is on the to-do-list since it is needed to create interactions with the player or other objects in the world.

Another thing I would like to turn my attention to is something I would most likely describe as [LOD](https://de.wikipedia.org/wiki/Level_of_Detail) system, but for simulations. Can you maybe detect stationary flow and replace it with a much cheaper version to allow much larger volumes of water? As you can imagine, most of the water will be stationary up until the player interacts with it. Or you can even go a step further and completely unload a whole area. This would allow for a system which can even be integrated into an open world game.

As you can see there are still many interesting directions yet to dive into and this is only the beginning.

### Further References
Here you can find some other resources I gathered regarding this topic to get you up and running with your research:

- <https://openreview.net/pdf?id=B1lDoJSYDH> _// The main paper described here: Lagrangian Fluid Simulation with Continuous Convolutions; Benjamin Ummenhofer, Lukas Prantl & Nils Thuerey, Vladlen Koltun_
- <https://openreview.net/forum?id=B1lDoJSYDH> _// Q&A with the authors and rewievers_
- <https://iclr.cc/virtual_2020/poster_B1lDoJSYDH.html> _// short 5 min presentation of the paper's authors at ICLR_
- <https://github.com/isl-org/DeepLagrangianFluids> _// GitHub repository of the paper_
- <https://physicsbaseddeeplearning.org> _// online book about combining physics and deep learning_
- <https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2023.0058>
- <https://arxiv.org/abs/2210.06036> _// Guaranteed Conservation of Momentum (extension of this method, same authors)_
- <https://arxiv.org/abs/2203.16797> _// A Survey of Physics-Informed Machine Learning_
- <https://inductiva.ai/blog/article/sph-1-a-smooth-motivation> _// blog post about SPH and graph based approaches_
- <https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_fluid-EulerParticle.pdf> _Fluid Simulation For Computer Graphics: A Tutorial in Grid Based and Particle Based Methods, Colin Braley, Adrian Sandu_
- <https://www.youtube.com/watch?v=rSKMYc1CQHE> // Sebastian Lague Coding Adventure: Simulating Fluids, visual great explanation of classical SPH simulators
- <https://www.youtube.com/watch?v=NmLK_WQBxB4> // MIT basics of machine learning
- <https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf> // Learning to Simulate Complex Physics with Graph Networks; Alvaro Sanchez-Gonzalez et al.; paper about a graph-based simulation method directly comparing itself to CConv
- <https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a-supp.pdf> // supplementary material with further discussion regarding differences to CConv
