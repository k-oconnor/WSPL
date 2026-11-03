# Related work and intended contribution

This is an initial positioning map, not an exhaustive novelty review. Primary sources were checked on November 2, 2026.

| Prior work | Established direction | Relevance / distinction for this project |
| --- | --- | --- |
| [De Giacomo & Vardi, LTLf/LDLf synthesis, IJCAI 2015](https://ijcai.org/proceedings/2015/223) | Finite-trace specifications can be translated to automata and strategies synthesized by distinguishing agent/environment propositions. | Use as formal foundation. A DFA alone does not supply a physical skill model or guarantee skill outcomes. |
| [Lang2LTL, CoRL 2023](https://arxiv.org/abs/2302.11649) | LLM-grounded natural-language commands translated to LTL for temporal robot navigation in unseen environments. | Language-to-temporal-logic robotics is prior work. Our choice of LTLf and weight-authored motor implementations needs its own evidence. |
| [Code as Policies](https://arxiv.org/abs/2209.07753) | Language models write robot policy programs, including feedback loops and primitive API composition. | A direct code-policy baseline is important. Skillforge asks whether fixed neural/RMV weights can serve as the executable skill artifact instead. |
| [SayCan](https://arxiv.org/abs/2204.01691) | High-level language knowledge is grounded through pretrained robotic skills and affordance/value information. | Skill availability/applicability and composition already have strong antecedents. Our acquisition-cost question concerns writing the skill implementation itself. |
| [Automatic encoding and repair of reactive high-level tasks with learned abstract representations](https://doi.org/10.1177/02783649231167207) | Skill abstractions support temporal specifications, robot behavior synthesis and suggestions for repairing unrealizable tasks. | Skill modeling, temporal synthesis and repair are not new as a combination. Compare the granularity and evidence for repair rather than claiming the combination itself. |
| [Sparks of Weight-Space Programming, November 2, 2026](https://rikiyat.github.io/blog/weight-space-programming/) | An early author report explores general-purpose LLMs reading, writing and editing neural weights; experiments vary in size and allowed feedback. | Motivates small weight-authored skills. It does not establish reliable robotic RMV controllers or their acquisition-cost advantage. |

The proposed contribution is a measured combination of fixed-interface weight-authored skills, faithful grounded RMV implementations, finite-trace composition and local revision. Test performance, interaction cost, wall time, fidelity, interpretability and retention separately. Do not claim that a generic LLM/LTL/skill hierarchy is novel, or that this is already a causal world model.

An especially relevant distinction in Pacheck et al. is that suggested skill additions/modifications still require a user to implement the physical skills. Generating and validating those implementations as grounded neural/RMV weights is a concrete direction to investigate, not an established novelty finding.

Further review is needed before publication: neuro-symbolic control, options and hierarchical reinforcement learning, contract-based control, task-and-motion planning, controller synthesis, certified neural control, fuzzy/MV logic networks, hypernetworks and program-to-network compilation. No priority claim has been established.
