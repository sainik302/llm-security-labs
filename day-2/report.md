## AI Threat Modeling – Day 4

### Potential Threats Identified:
- **Data Poisoning**: Malicious actors could insert malicious data into the training set, leading to biased or incorrect model predictions.
- **Model Inversion**: Attackers might try to reverse-engineer the model to retrieve private data used in training.
- **Evasion Attacks**: Adversarial examples that cause the model to misclassify legitimate inputs.
- **Inference Manipulation**: Manipulating the input to the model (e.g., prompt injection) to trigger unintended behavior.

### Model Vulnerabilities:
1. **Data Collection**:
   - Are there biases in the collected data?
   - Could attackers feed in biased or malicious data?
2. **Training**:
   - Are we using secure methods for training, like adversarial training?
   - Is the model resistant to data poisoning?
3. **Inference**:
   - Can attackers manipulate the model during inference (e.g., generating misleading input)?
