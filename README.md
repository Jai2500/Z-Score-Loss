# Z-Score-Loss
Repository for "Loss function to optimise signal significance in particle physics"

Please wait until the NeurIPS Workshop date (15th December) for the public release of the code and package. 

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

