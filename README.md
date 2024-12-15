# Repository for "Loss function to optimise signal significance in particle physics"

## We will be releasing the code late December; apologies for the delay. 

# Loss function to optimise signal significance in particle physics

[Jai Bardhan](https://jaibardhan.com/)<sup>1</sup> , [Cyrin Neeraj](https://inspirehep.net/authors/1904817)<sup>1</sup>, [Subhadip Mitra](https://sites.google.com/site/subhadipmitra/)<sup>1</sup>, [Tanumoy Mandal](https://scholar.google.co.in/citations?user=IwMWofEAAAAJ&hl=en)<sup>2</sup>   

<sup>1</sup> International Institute of Information Technology, Hyderabad, Telangana, IN.

<sup>2</sup> Indian Institute of Science Education and Research, Thiruvananthapuram, Kerala, IN.

At collider experiments at the LHC, beams of high energy particles are collided against one another to search for interesting New Physics phenomena signatures. Due to the rare nature of these NP phenomena, separating them from huge commonplace Standard Model background is a an extremely challenging task. Sophisticated deep learning models are used to separate interesting signal events from that of the SM background using simulated data and then, deployed on real collision data. The signal plus background hypothesis ($H_1$) is tested against the null or background-only hypothesis ($H_0$), and the disagreement between them is expressed in terms of a $p$ value. An equivalent interpretation of the $p$ value is the significance score ($Z$) defined such that a Gaussian-distributed variable found $Z$ standard deviations away from its mean has a tail-distribution probability equal to $p$. Most sensitivity studies commonly use a simple approximation of the median $Z$ score as a measure of the estimated signal significance,
$$Z \approx N_s/\sqrt{N_b}$$
where $N_s$ and $N_b$ are the estimated numbers of signal and background events, respectively. (In the rest of the paper, we shall refer to the median $Z$ score as just the $Z$ score.

Usually, a cross-entropy loss (BCE) or some weighted variants of it used to train classifiers to separate signal events from the backgrounds. Training then does not make use of the first-principle knowledge on the scattering rates of processes that are to be considered in a search; this can be obtained from quantum field theoretic calculations. Furthermore, BCE losses is less aligned to metric we perform the hypothesis testing with. 

Our motivations, are therefore, are two-fold:
1. Incorporating scattering rates of particular processes considered in a particle physics search systematically while training the classifier.
2. Usual losses for classification tasks (like BCE) that optimise the signal-to-background ratio ($r=N_s/N_b$) need not maximise the significance as the $Z$ score depends on the ratio and the absolute number (i.e., set size) of signal or background events ($Z\approx \sqrt{N_s}\cdot\sqrt{r} = r\sqrt{N_b}$).

Here we ask (and answer), can we derive a loss function that maximises the $Z$ score directly?

The $Z$ score  is a set function. Therefore, we define a surrogate loss function using the Lovàsz extension to maximise it directly. We evaluate this loss with pseudo-data mimicking a typical event classification task using a linear model and compare the decision boundaries to that model trained on a BCE loss. We also compare the performance of models trained on a BCE loss.

## Constructing the loss: key conditions 

We must consider some points before constructing a loss function based on the $Z$ score. First, since the $Z$ score is not a differentiable function (it depends upon discrete quantities), it needs a smooth interpolation. Second, the metric operates on datasets instead of individual samples---particularly, count data. Therefore, we must either develop a method to directly optimise the set function or assign contributions to specific samples within the set to optimise. 

We look for a smooth submodular function. A submodular function is a function that captures the concept of diminishing returns. It is defined on sets and has a property similar to concavity. Formally, submodularity can be defined as:

#### Definition: Submodularity

A set function $\Delta: \{0, 1\}^p \to \mathbb{R}$ is submodular if for all sets $A, B \in \{0, 1\}^p$

$$\Delta(A) + \Delta(B) \geq \Delta(A \cup B) + \Delta (A \cap B),$$

or equivalently $B \subseteq A$ and $i \notin A, i \notin B$,

$$\Delta(A \cup \{i\}) - \Delta(A) \leq \Delta(B \cup \{i\}) - \Delta(B).$$

The submodular functions can be optimised using greedy optimisation techniques, and it is to find optimal solutions in polynomial times. However, these discrete optimisation techniques cannot be used directly without a gradient. 

The Lovàsz extension allows us to associate a continuous, convex function with any submodular function. 

#### Definition: Lovàsz extension <a name=lovext></a>

For a set function $\Delta: \{0, 1\}^p \to \mathbb{R}$, the Lovàsz extension $\bar{\Delta}: [0, 1]^p \to \mathbb{R}$ is defined as

$$\bar{\Delta}: \mathbf{m} \in \mathbb{R}^p \mapsto%\rightarrowtail 
        \sum_{i=1}^p m_i\ g_i(\mathbf{m})$$

where $\mathbf{m} \in \mathbb{R}^p_+$ is the vector of errors (which we discuss in the next section), $g_i(\mathbf{m}) = \Delta(\{\pi_1, \dots, \pi_i\}) - \Delta(\{\pi_i, \dots, \pi_{i-1}\})$ and $\boldsymbol{\pi}$ is a permutation ordering the components of $\mathbf{m}$ in decreasing order, i.e., $x_{\pi_1} \geq x_{\pi_2} \geq \dots \geq x_{\pi_p}$

For the Lovàsz extension to be applicable, the set function must be submodular.

Additionally, the Lovàsz extension of a submodular function preserves submodularity, i.e., the extension evaluated at the points of the hypercube still follows submodularity.

## Building a loss from $Z$ score: Surrogate $\Delta_Z (y, \tilde{y})$

We propose the following set function, a surrogate to the $Z$ score and very similar in form, as the pre-cursor to the loss construction:

$$\Delta_Z(y, \tilde{y}) = \sum_{i \in S} \frac{\sigma_i \mathcal{L}}{\sqrt{\epsilon}} - \frac{\sum_{i \in S} \frac{v_i - n_i}{v_i} \sigma_i \mathcal{L}}{\sqrt{ \epsilon + \sum_{i \in B} \frac{p_i}{v_i} \sigma_i \mathcal{L}}},$$

where,
* $y$ $(\tilde{y})$ $\to$ ground-truth (predicted) label
* $\nu_i$ $\to$ number of events of process type $i$ ($i\in S \cup B$)
* $n_i, p_i$ $\to$ number of false negatives, false positives
* $\mathcal{L} \to$ Luminosity at which the experiment is performed.
* $\epsilon,\;\sum_{i \in S} \sigma_i \mathcal{L}/\sqrt{\epsilon}$ $\to$ added to ensure $\Delta_Z(\emptyset) = \boldsymbol{0}$.

In the [paper](https://arxiv.org/abs/2412.09500), we prove that the surrogate above is submodular (Appendix 1) and, hence, get a convex loss function, $\bar{\Delta}_Z (y,\tilde{y})$, to train classifiers using the Lovàsz extension.
#### Choice of error $\boldsymbol{m}$ (from [Lovàsz extenstion](#lovext))

We pick the error is given by the hinge loss, 
$$m_i = \max(1 - F_i(x)y_i, 0), \qquad y_i \in \{-1, 1\},$$
where, the labels are considered signed ($y_i \in \{-1, 1\}$) and the model outputs a score $F_i(x)$ for each sample $x$. 

### Parameters of $\bar{\Delta}_Z (y,\tilde{y})$

There is only one free parameter in our loss: $\epsilon$. Other quantities like $\sigma_i$ and $\mathcal{L}$ are set by the process under consideration (i.e., the particular classification task) and the collider experiment. Assuming we perform the classification for rare signals, we set $\epsilon = \sigma_{s}\mathcal{L}$, the theoretically predicted number of signal events (which is also the maximum number of estimated signal events) for testing the loss. 

## Testing the loss on pseudo-data

Our goal is to separate the signal ($s$) from background events using a linear classifier in the presence of multiple (say, two, $b1$ and $b2$) dominant background processes, as is usually the case. The datasets are modelled as normal distributions in two features, $x_1$ and $x_2$ which can be thought of as the kinematic features of the actual events. We generated roughly $50000$ points for each process and the optimisation was done in batches using ``RAdam`` optimiser. 

We train the linear classifier using the BCE loss and $\bar\Delta_Z$ with the hinge error for the following two test cases: 
* Case 1: $\sigma_{b1} = 1$ fb, $\sigma_{b2} = 100$ fb; $\sigma_{s} = 0.1$ fb.
* Case 2: $\sigma_{b1} = 100$ fb, $\sigma_{b2} = 1$ fb; $\sigma_{s} = 0.1$ fb.

We set the luminosity to $\mathcal{L}=3000$ fb$^{-1}$. 
<!---

![Case 1 - Decision Boundaries](https://github.com/Jai2500/Z-Score-Loss/blob/main/images/dec-bound-case1.png)
![Case 2 - Decision Boundaries](https://github.com/Jai2500/Z-Score-Loss/blob/main/images/dec-bound-case2.png)
-->

<img src="https://github.com/Jai2500/Z-Score-Loss/blob/main/images/dec-bound-case1.png" width="500"> <img src="https://github.com/Jai2500/Z-Score-Loss/blob/main/images/dec-bound-case2.png" width="500">


Left Panel: Decision boundaries for Case 1; Right Panel: Decision boundaries for Case 2.

Feature distributions are the same for both cases; the cases differ in the scattering rates of the processes considered.

The classifier trained with $\Delta_Z (y, \tilde{y})$ decision boundaries are scattering cross-section dependant compared to one trained with a BCE, as you can see from the figures above.  As $\bar\Delta_Z$ is designed to optimise the $Z$ score, eliminating more events from the larger background gives better significance scores. 

## Classification Metrics
<!---

![$Z$ score vs. linear model threshold](https://github.com/Jai2500/Z-Score-Loss/blob/main/images/Z-vs-thresh.png)

![$Z$ score vs. efficiency of signal events](https://github.com/Jai2500/Z-Score-Loss/blob/main/images/Z-vs-epsSig.png)

![Efficiency vs. linear model threshold](https://github.com/Jai2500/Z-Score-Loss/blob/main/images/eps-vs-thres.png)
-->
<img src="https://github.com/Jai2500/Z-Score-Loss/blob/main/images/Z-vs-thresh.png" width="500"> 

<img src="https://github.com/Jai2500/Z-Score-Loss/blob/main/images/Z-vs-epsSig.png" width="500">

<img src="https://github.com/Jai2500/Z-Score-Loss/blob/main/images/eps-vs-thres.png" width="500">

(From top left) First panel: The estimated $Z$ score for the entire range of the linear model threshold, $u$. 

Second panel: The distribution of the $Z$ score with signal efficiency, the fraction of signal events retained. Both quantities are functions of $u$. The model trained with $\bar{\Delta}_Z$ reaches the maximum $Z$ score for higher values of signal efficiency than that with the BCE loss. 

Third panel: Class efficiencies vs. $u$ when trained with the $\bar{\Delta}_Z$ loss with the hinge error. With increasing $u$, the larger background is eliminated first. For very high $u$ the drop in the subdominant background is less steeper than the signal, leading to the drop in $Z(u)$ in (a) for higher thresholds.

For the scans presented above, we demand $\varepsilon(u)\geq 0.05$ and the number of background events beyond the threshold to be at least $5$. Similar plots are obtained for Case 2 also. From the figure, we see that $\bar\Delta_z$ maximises the $Z$ score for a higher signal efficiency than the BCE, i.e., where the estimated $Z$ score peaks, the model retains more signal events than the BCE-trained model. (For the $\bar\Delta_z$ model, the estimated $Z$ score drops for high values of $u$ because there, for the datasets we consider, the major background is almost eliminated and further shifting the decision boundary reduces the minor background slower than the signal).

### ROC curves

We plot the ROC curves for experiments for Case 1. Case 2 gives similar results. Let $N_{B1}, N_{B2}$ represent the total number of $b1$ and $b2$ events generated in the dataset. Let $n_{b1}, n_{b2}$ represent the number of $b1$ and $b2$ events remaining after the threshold respectively. Let $\sigma_{b1}, \sigma_{b2}$ represent the cross sections of process $b1$ and $b2$ respectively. The total background efficiency is given by,

$$\frac{n_{b1} + n_{b2}}{N_{b1} + N_{b2}}$$

and the true background efficiency is given by,

$$\frac{\left(\frac{n_{b1}}{N_{b1}}\right) \sigma_{b1} + \left(\frac{n_{b2}}{N_{b2}}\right) \sigma_{b2}}{\sigma_{b1} + \sigma_{b2}}.$$

![ROC Curve for dataset (total) background efficiency vs signal efficiency.](https://github.com/Jai2500/Z-Score-Loss/blob/main/images/epsSig-vs-epsBG.png)

![ROC Curve for true background efficiency vs signal efficiency.](https://github.com/Jai2500/Z-Score-Loss/blob/main/images/epsSig-vs-weightedBGeff.png)

The true background efficiency differs from the total background efficiency in that it accounts for the cross sections of the background processes. ROC curves show that $\bar{\Delta}_Z(y, \tilde{y})$ performs better when we consider the true background distribution (i.e., accounting for the cross sections).

## Limitations, Scope
While our results are promising, further tests are needed to fully characterise and understand the benefits and limitations of $\bar \Delta_Z$.  Here, our choice of using a linear classifier on simple datasets was motivated by its simplicity and interpretability. However, for realistic characterisations, one has to look beyond the linear classifier (e.g., use a deep neural network) and consider a range of benchmark (new-physics) scenarios with different kinematics (features). For example, there could be multiple (more than two) major backgrounds with highly overlapping features or the signal size could be much smaller than the backgrounds (more than what we considered, as is the case in some heavy particle searches). 

Finally, we note that while it is possible to introduce rate-dependent weights directly in the BCE loss, tuning them is an empirical task. The weights that yield the best performance need not be simply the rates of the processes. In contrast, $\bar{\Delta}_Z$ presents a natural way to include the rates (cross sections) as it is derived from the significance score used in collider searches.

