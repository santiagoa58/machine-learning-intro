# Project Guidelines

This document defines the teaching philosophy, content standards, and structural requirements for all materials in this machine learning introduction project.

## Table of Contents

1. [Core Teaching Philosophy](#core-teaching-philosophy)
2. [Learning Science Principles](#learning-science-principles)
3. [Content Structure Requirements](#content-structure-requirements)
4. [Writing Style Guidelines](#writing-style-guidelines)
5. [Technical Accuracy Standards](#technical-accuracy-standards)
6. [Code Standards](#code-standards)
7. [Visual and Interactive Elements](#visual-and-interactive-elements)
8. [Review Checklist](#review-checklist)

---

## Core Teaching Philosophy

### The "Application to Theory" Approach

Every tutorial MUST follow this sequence:

1. **Application First** (Section 2)
   - Start with a real-world problem that's relatable and interesting
   - Show working code and results BEFORE explaining theory
   - Let learners see what the algorithm can do
   - Build intuition through hands-on experience

2. **Theory Second** (Section 3)
   - Only after seeing it work, explain how it works
   - Motivate every theoretical concept by connecting it back to the application
   - Use the application as a running example throughout theory
   - Mathematics should answer questions raised by the application

### Why This Approach?

**Traditional approach:** "Here's 10 pages of math, now let's see why it matters"
- Result: Learners are lost, demotivated, don't see the point

**Our approach:** "Here's what it does (cool!), now let's understand how (motivated!)"
- Result: Learners are engaged, curious, and ready to dive deeper

### Four Fundamental Principles

These principles apply to ALL work in this project. They are listed in order of priority:

#### 1. Simplicity & Clarity (HIGHEST PRIORITY)
- **Most important rule**: If a learner doesn't understand, we have failed
- Prefer clear explanation over technically precise jargon
- Break complex concepts into digestible pieces
- Use analogies and real-world comparisons
- Progressive complexity: simple → intermediate → advanced

#### 2. Absolute Correctness
- Technical accuracy is non-negotiable
- All mathematical formulations must be precise
- All code must be tested and verified
- Domain/range specifications must be exact
- Common misconceptions must be explicitly addressed

#### 3. Depth & Thoroughness
- Never gloss over important concepts
- Explain the "why" behind every "what"
- Provide complete explanations, not summaries
- Include edge cases and limitations
- Comprehensive, not superficial

#### 4. Limited Jargon
- Minimize technical jargon whenever possible
- When jargon is necessary, define it immediately and clearly
- Use plain language as the default
- Technical terms should be introduced, not assumed

---

## Learning Science Principles

**CRITICAL INSIGHT:** Research in cognitive psychology shows that passive reading creates an "illusion of competence"—learners feel they understand while reading but struggle to apply knowledge independently. Our materials must engage learners in active cognitive processes that build durable, transferable understanding.

This section incorporates evidence-based principles from learning science research. For complete research citations and detailed analysis, see **LEARNING_SCIENCE_REVIEW.md**.

### Active Learning Over Passive Consumption

**The Problem with Passive Learning:**
- Reading and observing code feels easy → feels like learning
- But easy = weak memory formation
- Result: Rapid forgetting, poor transfer to new problems

**The Solution:**
Every tutorial must include active engagement opportunities. Learners should:
- **Retrieve** information from memory (not just recognize it)
- **Generate** answers before seeing solutions
- **Apply** concepts to new situations
- **Reflect** on their understanding

### Five Essential Learning Science Principles

#### Principle 1: Retrieval Practice (CRITICAL)

**Research Finding:** Testing yourself is more effective than re-reading for long-term retention (Roediger & Karpicke, 2006).

**Implementation Requirements:**

Every tutorial MUST include:

1. **"Quick Check" sections** (3-5 per tutorial)
   - After each major concept or section
   - 2-4 questions requiring recall without scrolling back
   - Use `<details>` tags to hide answers
   - Mix factual and conceptual questions

2. **End-of-tutorial assessment**
   - Comprehensive practice problems
   - Progressive difficulty: guided → independent
   - Both application and theory questions

**Example:**
```markdown
### 🧠 Quick Check

Try to answer without scrolling back:

**1. What are the two inputs to train_test_split()?**
<details>
<summary>Answer</summary>
Features (X) and targets (y)
</details>

**2. Why do we split data instead of training on everything?**
<details>
<summary>Answer</summary>
To evaluate performance on unseen data and detect overfitting
</details>
```

**Why This Works:** The act of retrieving information strengthens memory more than passive review. Even failed retrieval attempts improve subsequent learning.

#### Principle 2: Worked Examples with Faded Guidance

**Research Finding:** Learners acquire skills most effectively through a progression from complete examples to independent problem-solving (Sweller et al., 1998).

**Implementation Requirements:**

For every complex skill, provide this 4-step progression:

1. **Step 1: Fully worked example** (current standard ✓)
   - Complete code with detailed explanation
   - Show every step

2. **Step 2: Completion problem** (MUST ADD)
   - Same problem with 20-40% blanks to fill
   - Provide hints
   - Hide solution in `<details>` tag

3. **Step 3: Guided problem** (MUST ADD)
   - Similar problem, different context
   - Hints available but no code scaffolding
   - Learner writes from scratch with support

4. **Step 4: Independent problem** (MUST ADD)
   - Novel application requiring transfer
   - No hints, no solution provided
   - Learner works completely independently

**Example Structure:**
```markdown
### Worked Example: Linear Regression
[Complete code with full explanation]

### 🔨 Your Turn: Complete This
```python
def gradient_descent(...):
    y_pred = _____  # Hint: mx + b
    gradient = _____  # Hint: derivative of cost
```

### 🚀 Similar Problem
Apply to housing prices dataset (hints provided)

### 💪 Independent Challenge
Apply to your own dataset (no support)
```

**Why This Works:** Gradually removing support prevents cognitive overload while building independent capability.

#### Principle 3: Metacognition - Think About Thinking

**Research Finding:** Learners who monitor and regulate their understanding achieve better outcomes (Schraw & Dennison, 1994).

**Implementation Requirements:**

Every tutorial MUST include:

1. **Learning objectives before complex sections**
   ```markdown
   ### 📍 Before We Begin: Gradient Descent

   After this section, you should be able to:
   - Explain why we need optimization
   - Calculate one gradient descent step
   - Identify when it might fail
   ```

2. **Reflection checkpoints after complex sections**
   ```markdown
   ### 🤔 Reflection

   Take 2 minutes:
   1. What was the main insight?
   2. What's still confusing?
   3. Can you explain this to a friend?

   Rate your understanding (1-5): [ ]
   If below 3, review the section
   ```

3. **Self-assessment opportunities**
   - Confidence ratings
   - Self-explanation prompts
   - "What questions do you still have?"

**Why This Works:** Metacognitive awareness helps learners identify gaps, allocate study time effectively, and develop self-directed learning skills.

#### Principle 4: Address Misconceptions Explicitly

**Research Finding:** Misconceptions persist unless directly confronted and replaced with correct understanding.

**Implementation Requirements:**

Every tutorial MUST include 3-5 "Common Misconceptions" boxes:

```markdown
### ⚠️ Common Misconceptions

**Misconception 1: "R² always ranges from 0 to 1"**
- ❌ **Why it's wrong:** R² can be negative!
- ✅ **Correct:** R² ranges from -∞ to 1. Negative means worse than baseline.
- 🧠 **Check:** If R²=-0.5, what does that mean?
  <details><summary>Answer</summary>
  Model performs worse than just predicting the mean
  </details>
```

**Required Elements:**
- State the misconception explicitly
- Explain why it's tempting/wrong
- Provide correct understanding
- Give concrete example showing the difference
- Include self-check question

**Why This Works:** Merely presenting correct information doesn't erase misconceptions. They must be explicitly identified and refuted.

#### Principle 5: Spaced Repetition and Interleaving

**Research Finding:** Distributing practice over time with mixing of topics improves long-term retention by 20-30% (Cepeda et al., 2006).

**Implementation Requirements:**

1. **Within tutorials:** Revisit concepts
   - Reference earlier concepts when introducing new ones
   - "Recall from Linear Regression that we split data..."
   - Build cumulative understanding

2. **Across tutorials:** Interleave content
   - KNN tutorial should review train/test split from Linear Regression
   - Later tutorials mix old and new concepts
   - Create cumulative review notebooks

3. **Spaced retrieval schedules:**
   ```markdown
   ### 🔄 Spaced Review

   To maximize retention:
   - ✅ Today: Complete all challenges
   - 📅 Tomorrow: 5-minute quiz
   - 📅 Next week: Mixed practice
   - 📅 Next month: Cumulative review
   ```

**Why This Works:** Spacing creates "desirable difficulties" that strengthen memory. Interleaving improves discrimination between concepts and transfer.

### Additional Evidence-Based Techniques

#### Prediction Prompts (Generation Effect)

Before revealing answers, ask learners to predict:

```markdown
### 🔮 Before Running Code

Predict:
1. Will training or test accuracy be higher?
2. What R² range do you expect?

**Now run and compare to your prediction**

Reflection:
- Were you close?
- What surprised you?
```

**Why This Works:** Generating an answer (even wrong) before seeing the correct one improves learning more than passive reading.

#### Concrete-Representational-Abstract (CRA) Sequence

For complex concepts, use this progression:

1. **Concrete:** Physical analogy or real-world example
2. **Representational:** Visual diagram or animation
3. **Abstract:** Mathematical formula or general principle

**Example: Gradient Descent**
- Concrete: "Blindfolded on a hill, feeling for downslope"
- Representational: Animated visualization of ball rolling down surface
- Abstract: $m := m - \alpha \frac{\partial J}{\partial m}$

#### Elaborative Interrogation ("Why" Questions)

Prompt learners to explain why things work:

```markdown
### 🤔 Deep Question

Why do we square the errors in MSE instead of using absolute values?

Try to think of 3 reasons before revealing:
1. ______
2. ______
3. ______

<details>
<summary>Expert Reasoning</summary>

1. Mathematical: Squaring is differentiable; absolute value isn't
2. Conceptual: Penalizes large errors more heavily
3. Statistical: Relates to variance
4. Practical: One large error worse than two small ones

</details>
```

### Balancing Cognitive Load

**Working Memory Limitations:**
- Humans can hold ~4 chunks of information in working memory
- Excessive load prevents learning

**Guidelines:**

1. **Chunk information meaningfully**
   - Group related concepts
   - Use the "What, Why, How" pattern
   - Break complex derivations into steps

2. **Reduce extraneous load**
   - Clear formatting and organization
   - Minimize distracting elements
   - Consistent structure across tutorials

3. **Manage intrinsic load**
   - Start simple, build complexity gradually
   - Use the application-to-theory approach
   - Provide prerequisites clearly

4. **Optimize germane load**
   - Focus attention on schema building
   - Connect new to prior knowledge
   - Make patterns explicit

### Motivation and Self-Efficacy

**Research Finding:** Self-efficacy (belief in one's ability) predicts learning outcomes better than past achievement (Bandura).

**Implementation Strategies:**

1. **Normalize struggle**
   ```markdown
   ### 💪 Feeling Confused?

   That's completely normal! Gradient descent is challenging.
   Even ML experts found this hard at first.

   Struggle is a sign your brain is growing.
   ```

2. **Provide early wins**
   - Start with achievable challenges
   - Celebrate progress
   - Build confidence progressively

3. **Growth mindset messaging**
   - Emphasize effort over innate ability
   - Frame mistakes as learning opportunities
   - Provide specific strategies when stuck

4. **Real-world relevance**
   - Show why concepts matter
   - Connect to career applications
   - Highlight practical impact

### Quality Checklist for Learning Science Integration

Before releasing any tutorial, verify:

**Active Learning:**
- [ ] 3-5 "Quick Check" retrieval sections
- [ ] End-of-tutorial practice challenges
- [ ] Prediction prompts before reveals

**Worked Examples:**
- [ ] Complete example → Completion → Guided → Independent progression
- [ ] Faded guidance for complex skills
- [ ] Multiple practice opportunities

**Metacognition:**
- [ ] Learning objectives before complex sections
- [ ] Reflection checkpoints after complex sections
- [ ] Self-assessment throughout

**Misconceptions:**
- [ ] 3-5 common misconceptions explicitly addressed
- [ ] Explain why misconception is tempting
- [ ] Provide concrete corrective examples

**Spacing & Interleaving:**
- [ ] Concepts revisited within tutorial
- [ ] References to previous tutorials
- [ ] Spaced review schedule provided

**Motivation:**
- [ ] Struggle normalized with support strategies
- [ ] Early achievable challenges
- [ ] Growth mindset messaging
- [ ] Real-world relevance emphasized

### Implementation Priority

**Phase 1: Critical (Do First)**
1. Add retrieval practice (highest impact per hour invested)
2. Address misconceptions explicitly
3. Add end-of-tutorial challenges

**Phase 2: High Value (Do Second)**
4. Add learning objectives and reflection checkpoints
5. Implement worked example progressions
6. Add prediction prompts

**Phase 3: Enhancing (Do Third)**
7. Build spaced repetition system
8. Add multiple contexts for transfer
9. Implement CRA sequences for complex topics

### Measuring Effectiveness

Track these metrics to verify improvements:

**Learning Outcomes:**
- Quiz performance (immediate and delayed)
- Transfer task success rate
- Completion rates
- Time to competency

**Learner Experience:**
- Self-reported understanding
- Confidence ratings
- Satisfaction scores

**Expected Improvements with Full Implementation:**
- +20-30% on delayed retention tests
- +15-25% on transfer tasks
- +10-15% completion rates
- +25-35% self-reported confidence

---

## Content Structure Requirements

### Mandatory Structure

Every algorithm tutorial MUST follow this exact structure:

```
# [Algorithm Name]: From Application to Theory

## Table of Contents
[Generated automatically from sections]

## 1. Introduction
- Brief overview (2-3 paragraphs)
- What problem does this algorithm solve?
- Real-world use cases
- What will you learn?

## 2. Application
[THE BULK OF INITIAL LEARNING HAPPENS HERE]

### 2.1 Context and Disclaimer
- Introduce the specific problem we're solving
- Set expectations about the dataset
- Acknowledge limitations

### 2.2 Data Collection
- How to get/load the data
- What the data represents
- First look at the data structure

### 2.3 Data Preparation
- Cleaning and preprocessing
- Splitting into features and targets
- Train/test split

### 2.4 Building the Model
- Creating the model instance
- Explain parameters (even if using defaults)
- Why this algorithm for this problem?

### 2.5 Training the Model
- Fitting the model
- What happens during training?
- Computational considerations

### 2.6 Making Predictions
- How to use the trained model
- Interpreting predictions
- Examples of predictions

### 2.7 Evaluation
- Appropriate metrics for this problem
- Visualizing results
- What makes a good result?
- Limitations and when the model fails

## 3. Theory
[DEEP DIVE AFTER APPLICATION]

### 3.1 Introduction to Theory
- Transition from application
- "Now that you've seen it work, let's understand how"
- Roadmap of theoretical concepts

### 3.2+ [Algorithm-Specific Theory]
- Mathematical foundations
- Step-by-step derivations
- Intuitive explanations alongside math
- Connect back to application frequently

### 3.X Cost/Loss Function
- Why we need it
- Mathematical formulation
- Intuitive understanding

### 3.Y Optimization
- How parameters are learned
- Gradient descent or other methods
- Convergence and stopping criteria

### 3.Z Advanced Topics
- Extensions and variations
- Theoretical guarantees
- Computational complexity

## 4. Conclusion
- Recap of what was learned
- Key takeaways
- When to use this algorithm
- Links to related algorithms
```

### Section Numbering

- **Section 1**: Introduction
- **Section 2**: Application (practical, hands-on)
- **Section 3**: Theory (mathematical, conceptual)
- **Section 4**: Conclusion and next steps

This numbering is MANDATORY and consistent across all tutorials.

---

## Writing Style Guidelines

### Tone and Voice

- **Conversational but professional**
  - ✅ "Let's see what happens when we run this code"
  - ❌ "The subsequent execution of the aforementioned code block"

- **Encouraging and inclusive**
  - ✅ "This might seem complex at first, but let's break it down"
  - ❌ "Obviously, this is straightforward"

- **Direct and active**
  - ✅ "We split the data into training and testing sets"
  - ❌ "The data is split into training and testing sets"

### Explaining Concepts

#### The "What, Why, How" Pattern

For every new concept, answer in this order:

1. **What** - Define it in plain language
2. **Why** - Why do we need this? What problem does it solve?
3. **How** - How does it work? What's the implementation?

Example:
```markdown
### Train/Test Split

**What**: Dividing your dataset into two parts - one for training, one for testing.

**Why**: We need to test our model on data it has never seen before to ensure it's
actually learning patterns, not just memorizing the training data.

**How**: We use sklearn's train_test_split function, which randomly shuffles and
divides the data according to the ratio we specify.
```

#### Introducing Mathematics

- **Always motivate before formulating**
  - ❌ "The cost function is: J(θ) = ..."
  - ✅ "We need a way to measure how wrong our predictions are. This measurement is called a cost function: J(θ) = ..."

- **Explain every symbol**
  ```markdown
  The equation is:

  $$ y = mx + b $$

  Where:
  - $y$ is the predicted value
  - $m$ is the slope of the line
  - $x$ is the input feature
  - $b$ is the y-intercept
  ```

- **Provide intuition alongside formulas**
  ```markdown
  $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

  In plain English: we take each prediction error, square it (to make negatives positive
  and to penalize large errors more), then average all these squared errors.
  ```

### Avoiding Common Pitfalls

❌ **Don't assume knowledge**
- Bad: "Using standard gradient descent"
- Good: "Using gradient descent, an optimization algorithm that we'll explore in detail in Section 3.6"

❌ **Don't skip steps**
- Bad: "Therefore, the optimal parameters are..."
- Good: "Let's work through this step by step: First, we... Then we... Therefore..."

❌ **Don't use undefined acronyms**
- Bad: "Use MSE for regression tasks"
- Good: "Use MSE (Mean Squared Error) for regression tasks"

✅ **Do provide context**
- "In our stock price prediction example, this means..."
- "Going back to the Iris dataset we loaded earlier..."

✅ **Do acknowledge complexity**
- "This is the most mathematically intensive part, so take your time"
- "If this seems confusing, that's normal - we'll clarify with an example"

---

## Technical Accuracy Standards

### Mathematical Precision

1. **Domain and Range**
   - Always specify domain and range for functions
   - Use correct interval notation: (0, 1) vs [0, 1]
   - Example: "The sigmoid function has domain (-∞, ∞) and range (0, 1)"

2. **Asymptotic Behavior**
   - Clarify when functions approach but never reach values
   - Example: "The sigmoid approaches 0 and 1 asymptotically but never actually reaches these values"

3. **Complexity Notation**
   - Use Big-O notation correctly
   - Specify what variables represent: "O(N·d) where N is samples and d is dimensions"

4. **Statistical Metrics**
   - Specify bounds correctly: "R² ranges from -∞ to 1, not 0 to 1"
   - Explain special cases: "R² can be negative when the model performs worse than predicting the mean"

### Common Technical Mistakes to Avoid

❌ **Imprecise range descriptions**
- Bad: "Sigmoid outputs values between 0 and 1"
- Good: "Sigmoid outputs values in the open interval (0, 1), approaching but never reaching 0 or 1"

❌ **Incorrect complexity**
- Bad: "KNN has O(N²) complexity"
- Good: "Brute-force KNN has O(N·d) complexity for a single prediction, where N is training samples and d is dimensions"

❌ **Ambiguous array indexing**
- Bad: `cm[0][0] # True Negative`
- Good: `cm[0][0] # True Negative (assuming class 0 is negative - sklearn sorts labels)`

### Code Accuracy

1. **Reproducibility**
   - Always set `random_state` in stochastic operations
   - Example: `train_test_split(X, y, test_size=0.2, random_state=42)`

2. **Correct Function Usage**
   - Verify function signatures match actual usage
   - Include all necessary parameters
   - Document assumptions

3. **Error Handling**
   - Code should handle edge cases appropriately
   - Include comments for non-obvious behavior

---

## Code Standards

### Code Quality

1. **Readability**
   ```python
   # ✅ GOOD - Clear variable names, commented
   # Split data: 80% training, 20% testing
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # ❌ BAD - Unclear, no context
   a, b, c, d = train_test_split(X, y, test_size=0.2)
   ```

2. **Comments**
   - Explain WHY, not WHAT
   - ✅ "# Use log scale because feature values span many orders of magnitude"
   - ❌ "# Convert to log"

3. **Structure**
   - Imports at the top of each section where first needed
   - Explain new imports when introduced
   - Logical grouping of related code

### Code Explanations

Every code cell should be:
1. **Preceded by explanation** - What we're about to do and why
2. **Followed by interpretation** - What the output means

Example:
```markdown
Now let's train our model on the training data. Training means the model will
learn the patterns in the data by adjusting its internal parameters.

[CODE CELL]

The model has now learned from 80% of our data. Next, we'll test it on the
remaining 20% that it hasn't seen before.
```

---

## Visual and Interactive Elements

### Required Visualizations

1. **Data Exploration**
   - Always visualize the data before modeling
   - Use scatter plots, histograms, or appropriate visualization
   - Label axes clearly
   - Include titles that explain what you're showing

2. **Results Visualization**
   - Show predictions vs actual values
   - Include decision boundaries for classification
   - Use color coding effectively
   - Add legends and labels

3. **Performance Metrics**
   - Visualize training curves when relevant
   - Show confusion matrices for classification
   - Use appropriate scale (linear vs log)

### Visualization Standards

```python
# ✅ GOOD - Complete labeling and context
import matplotlib.pyplot as plt

plt.scatter(X, y, alpha=0.6, label='Actual Data')
plt.plot(X, predictions, color='red', linewidth=2, label='Model Predictions')
plt.xlabel('Day Number')
plt.ylabel('Stock Price ($)')
plt.title('Linear Regression: Stock Price Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ❌ BAD - No labels, unclear
plt.scatter(X, y)
plt.plot(X, predictions)
plt.show()
```

### Interactive Elements

Where possible, include:
- Multiple examples with different parameters
- Comparison of different approaches
- What-if scenarios
- Failure cases and limitations

---

## Review Checklist

Before submitting any content, verify ALL of the following:

### Structure Review
- [ ] Follows "Application to Theory" structure
- [ ] Section 1: Introduction is clear and motivating
- [ ] Section 2: Application is complete and working
- [ ] Section 3: Theory is thorough and well-motivated
- [ ] Table of contents is present and accurate
- [ ] All section numbers are consistent

### Content Review
- [ ] Every technical term is defined when first introduced
- [ ] Every mathematical formula has an intuitive explanation
- [ ] Every symbol in equations is explained
- [ ] Code examples work correctly
- [ ] All visualizations have proper labels and titles
- [ ] Links to other sections/notebooks work correctly

### Technical Accuracy Review
- [ ] All mathematical statements are precise
- [ ] Domain and range specifications are correct
- [ ] Complexity analysis is accurate
- [ ] Statistical metrics are correctly bounded
- [ ] Code includes `random_state` where needed
- [ ] Array indexing assumptions are documented
- [ ] No common misconceptions are present

### Style Review
- [ ] Writing is clear and accessible
- [ ] Tone is conversational but professional
- [ ] Jargon is minimized and explained
- [ ] "What, Why, How" pattern is followed
- [ ] Active voice is used
- [ ] Concepts build progressively

### Code Review
- [ ] Variable names are descriptive
- [ ] Comments explain WHY, not WHAT
- [ ] Code is preceded by explanation
- [ ] Output is interpreted and explained
- [ ] Edge cases are handled or documented
- [ ] Imports are explained when first introduced

### Completeness Review
- [ ] No "TODO" or placeholder sections
- [ ] All referenced sections exist
- [ ] All figures and images display correctly
- [ ] Examples are complete and tested
- [ ] Conclusion summarizes key points

---

## Examples of Good vs Bad Content

### Example 1: Introducing a Concept

❌ **BAD**
```markdown
### Mean Squared Error

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

This is the cost function we minimize.
```

✅ **GOOD**
```markdown
### Mean Squared Error (MSE)

We need a way to measure how wrong our predictions are. One popular approach is
Mean Squared Error (MSE).

**What it does**: MSE calculates the average of all squared prediction errors.

**Why square the errors**:
- Makes negative errors positive (a prediction that's 5 too high is just as
  bad as 5 too low)
- Penalizes large errors more than small ones (being off by 10 is more than
  twice as bad as being off by 5)

**The Formula**:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

Where:
- $n$ is the number of predictions
- $y_i$ is the actual value for example $i$
- $\hat{y}_i$ is our predicted value for example $i$
- $(y_i - \hat{y}_i)$ is the prediction error

**In plain English**: For each prediction, calculate how far off you were, square
that error, then average all the squared errors together.

In our stock price example, if we predicted $100 but the actual price was $105,
the error would be -5, and the squared error would be 25.
```

### Example 2: Code Introduction

❌ **BAD**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

✅ **GOOD**
```markdown
Now that we have our features (X) and targets (y), we need to split them into
training and testing sets. This is a crucial step in machine learning.

**Why split the data?**
We want to train our model on one portion of the data and test it on data it
has never seen before. This tells us if the model is actually learning patterns
or just memorizing the training data.

We'll use sklearn's `train_test_split` function, which randomly divides our
data. We'll use 80% for training and 20% for testing.

[CODE CELL]
```python
from sklearn.model_selection import train_test_split

# Split: 80% training, 20% testing
# random_state=42 ensures we get the same split every time (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```
```markdown
Now we have:
- `X_train` and `y_train`: 80% of the data for training
- `X_test` and `y_test`: 20% of the data for testing (model has never seen this)
```

---

## Maintaining These Standards

### For New Content

1. Review this document before starting
2. Use existing tutorials (Linear Regression, KNN) as templates
3. Follow the checklist before submitting
4. Ask: "Would I understand this if I knew nothing about ML?"

### For Content Updates

1. Ensure changes maintain the application-to-theory structure
2. Don't sacrifice clarity for brevity
3. Don't sacrifice accuracy for simplicity
4. Find the balance: accurate AND understandable

### Priority Order When Conflicts Arise

If you must choose between competing concerns, prioritize in this order:

1. **Correctness** - Never sacrifice technical accuracy
2. **Clarity** - If correct but unclear, simplify the explanation (not the accuracy)
3. **Completeness** - Better to be thorough than brief
4. **Consistency** - Follow established patterns

---

## Conclusion

These guidelines exist to ensure every learner, regardless of background, can
understand and apply machine learning concepts. When in doubt, ask:

- "Would this make sense to someone learning ML for the first time?"
- "Am I showing them why this matters before diving into how it works?"
- "Is every statement technically accurate?"
- "Have I explained every piece of jargon?"

Remember: **If the learner doesn't understand, we haven't taught effectively.**

The goal is not to impress with complexity, but to illuminate with clarity.
