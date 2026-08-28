### Base ML class usage documentation
Discription for endpoint for  BaseML class.
With given breef documentation we can make new ML Models using baseML object

updated at : [28/08/2026] : [1:49pm]

**BaseML**
parameters{
    hyper_para : {dictionary with value for respective unit}
    weights : shape(d,1) | default convention internal computed value
    data : shape(n,d) | main data matrix conventioned lock
    target : (n,1) | locked vectoric convention
}

hyper_para : 
    Requires internal documented rule with specific class with its conventions 
    need to stricly follow the convention while writing main training loop function
weights : 
    Documented conventional shape with (d,1) used same in prediction function
