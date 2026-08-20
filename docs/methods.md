# Methods

The proposed framework for intervention design on graph-structured data consists of three primary components: (1) a graph-based predictive model enhanced with dimensionality reduction, (2) a local counterfactual search to identify node-specific interventions, and (3) a global optimization phase to select a robust set of intervention policies.

## Proposed Predictive Architecture

To model the risk associated with individual nodes in a networked environment, we employ a Graph Convolutional Network (GCN). Given the high-dimensional nature of the input features $X \in \mathbb{R}^{d}$, we incorporate a pre-determined embedding layer based on Principal Component Analysis (PCA) to handle feature complexity. 

The architecture consists of a fixed linear projection layer $W_{PCA} \in \mathbb{R}^{d \times d'}$, where $d' \ll d$, followed by multiple graph convolutional layers. The PCA components are pre-calculated and remain non-trainable to ensure that the mapping between the reduced embedding space and the original features remains deterministic. This preservation is critical for the interpretability of counterfactual explanations, as it allows for the precise mapping of perturbations in the original feature space to changes in the model's predictions. The propagation rule for the $l$-th GCN layer is defined as:

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

where $\tilde{A}$ is the adjacency matrix with self-loops, $\tilde{D}$ is the corresponding degree matrix, and $H^{(0)}$ is the PCA-projected feature matrix. This GCN component is modular and can be substituted with any modern Graph Neural Network (GNN) variant, such as GraphSAGE or Graph Attention Networks (GAT), depending on the density and type of the graph data.

## Generating Local Counterfactuals

For each node $v$ correctly predicted as "high-risk" (positive class), we generate a local counterfactual explanation. A local counterfactual is a minimal set of interventions $c_v = \{i_1, i_2, \dots, i_k\}$ that, when applied to node $v$, flips the model's prediction from positive to negative.

### Neighborhood Feature Inclusion
Unlike traditional counterfactual methods that focus solely on the target node's features, our implementation explores the graph's topology. The search space for interventions is governed by an `edge_mode` parameter:
- **Feature Only (`none`)**: Perturbations are restricted to the primary features of node $v$.
- **Edge Modification (`edges`)**: The algorithm considers adding or removing edges between $v$ and other nodes in the graph. These topological changes are dynamically evaluated and converted into neighborhood feature distribution requirements.
- **Neighbor Feature Perturbation (`edge_features`)**: The algorithm considers modifying the features of $v$'s immediate neighbors to observe the influence of peer effects on $v$'s risk.

### Greedy Search and Condition Conversion
We employ an iterative greedy search (Algorithm 1) to find the counterfactual. In each step, we select the intervention (feature flip, edge change, or neighbor modification) that maximizes the reduction in predicted probability. 

Crucially, once a set of topological or neighbor changes is found, they are converted into **Neighbor Conditions** to ensure the explanation remains interpretable and generalizable. If an intervention modified $v$'s neighborhood (through edge changes or neighbor feature flips), we compute the resulting change in the mean feature distribution of the 1-hop neighborhood. This is expressed as a threshold condition:
$$N(\text{feature}_j) \in [\text{op}, \text{threshold}]$$
where $N(\cdot)$ denotes the neighborhood mean, and $\text{op} \in \{ \geq, \leq \}$. For example, adding a supportive peer might be converted to a requirement that the mean "social support" feature in the neighborhood exceeds a specific threshold. This abstraction allows the local policy to be applied and evaluated on other nodes.

### Algorithm 1: Local Counterfactual Generation
---
**Input:** Graph $G$, node $v$, GNN model $f$, maximum interventions $K$, `edge_mode`  
**Output:** Local counterfactual clause $c_v$

1. Initialize $c_v \leftarrow \emptyset$, current graph state $G_{mod} \leftarrow G$
2. **While** $|c_v| < K$ **and** $p = f(v, G_{mod}) > 0.5$:
3. &nbsp;&nbsp;&nbsp;&nbsp;Identify candidate interventions $I$ based on `edge_mode`
4. &nbsp;&nbsp;&nbsp;&nbsp;Select $i^* = \arg\max_{i \in I} [f(v, G_{mod}) - f(v, G_{mod} \oplus \{i\})]$
5. &nbsp;&nbsp;&nbsp;&nbsp;**If** marginal improvement $> 0$:
6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$c_v \leftarrow c_v \cup \{i^*\}$, $G_{mod} \leftarrow G_{mod} \oplus \{i^*\}$
7. &nbsp;&nbsp;&nbsp;&nbsp;**Else**: Break
8. Convert all neighbor-specific modifications in $c_v$ to threshold-based neighbor conditions
9. **Return** $c_v$
---

## Generating Global Counterfactuals

The global intervention strategy is derived by selecting a subset of local counterfactual clauses to form a Disjunctive Normal Form (DNF) policy that covers the maximum number of high-risk nodes.

### Problem Formulation
Let $P \subseteq V$ be the set of nodes predicted as positive. Let $\mathcal{C} = \{c_1, \dots, c_{|P|}\}$ be the set of candidate counterfactual clauses generated for each node in $P$. We seek a subset $S \subseteq \mathcal{C}$ that maximizes a global coverage function $F(S)$:
$$\max_{S \subseteq \mathcal{C}} F(S) = |\{v \in P : \exists c \in S \text{ s.t. } \text{Covered}(v, c)\}|$$
subject to a total cost (complexity) constraint:
$$\sum_{c \in S} |c| \leq B$$
where $|c|$ is the number of individual interventions within clause $c$.

### Coverage and Compatibility
A clause $c$ is said to "cover" a node $v$ if applying $c$ to $v$ is sufficient to flip $v$'s classification. Our implementation uses a two-tier evaluation for $\text{Covered}(v, c)$:
1. **Direct Flip**: Apply the self-feature interventions of $c$ to node $v$. If $f(v, G \oplus c_{feat}) \leq 0.5$, $v$ is covered.
2. **Rule Compatibility**: If the direct flip does not occur, $v$ is still covered if it satisfies its own minimal requirements $c_v$ via $c$. This requires that $c$ contains the identical self-feature modifications as $c_v$, and $c$'s neighbor conditions are *stricter* than or equal to those required by $c_v$ (e.g., $c$ requires $N(X) \geq 0.8$ while $c_v$ only requires $N(X) \geq 0.5$).

### Monotonicity and Submodularity
The global coverage function $F(S)$ possesses two critical mathematical properties:
- **Monotonicity**: $F(A) \leq F(B)$ for all $A \subseteq B \subseteq \mathcal{C}$. Adding more intervention policies to the strategy set never reduces the total number of individuals helped.
- **Submodularity**: $F(S)$ exhibits diminishing marginal returns, meaning the benefit of adding a clause $c$ to a small set of interventions is greater than or equal to the benefit of adding it to a larger set: $F(A \cup \{c\}) - F(A) \geq F(B \cup \{c\}) - F(B)$ for $A \subseteq B$.

### Greedy Strategy and Approximation
Given the NP-hard nature of the knapsack-constrained submodular maximization, we employ a greedy algorithm (Algorithm 2). The algorithm iteratively selects the clause $c^*$ that maximizes the marginal coverage gain per unit cost:
$$c^* = \arg\max_{c \in \mathcal{C} \setminus S} \frac{F(S \cup \{c\}) - F(S)}{|c|}$$
This greedy strategy is guaranteed to achieve a **1/2-approximation** of the optimal solution under the knapsack constraint.

### Algorithm 2: Greedy Global Intervention Selection
---
**Input:** Local counterfactual candidates $\mathcal{C}$, budget $B$, coverage function $F$  
**Output:** Global intervention strategy $S$

1. Initialize $S \leftarrow \emptyset$, total_cost $\leftarrow 0$
2. **While** clauses in $\mathcal{C} \setminus S$ remain:
3. &nbsp;&nbsp;&nbsp;&nbsp;Select $c^* = \arg \max_{c \in \mathcal{C} \setminus S} \frac{F(S \cup \{c\}) - F(S)}{|c|}$
4. &nbsp;&nbsp;&nbsp;&nbsp;**If** total_cost $+ |c^*| \le B$:
5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$S \leftarrow S \cup \{c^*\}$, total_cost $\leftarrow$ total_cost $+ |c^*|$
6. &nbsp;&nbsp;&nbsp;&nbsp;**Else**: Remove $c^*$ from consideration
7. **Return** $S$
---


### Intervention Hypotheses
By restricting the counterfactual search space to only mutable features (e.g., career intent, social participation) and excluding immutable traits (e.g., age, race), the resulting global strategy $S$ serves as a set of **Intervention Hypotheses**. These hypotheses provide actionable, evidence-based recommendations for systemic risk mitigation in networked populations.
