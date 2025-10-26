import type { Metadata } from 'next';
import { SidebarLayoutContent } from '@/components/layout/sidebar-layout';
import {
  Breadcrumbs,
  BreadcrumbHome,
  BreadcrumbSeparator,
  Breadcrumb,
} from '@/components/layout/breadcrumbs';

export const metadata: Metadata = {
  title: "Project Guidelines",
  description: "Teaching philosophy, content standards, and learning science principles for creating effective ML tutorials.",
  openGraph: {
    title: "Project Guidelines | ML Introduction",
    description: "Teaching philosophy, content standards, and learning science principles for creating effective ML tutorials.",
    type: "article",
  },
};

export default function GuidelinesPage() {
  return (
    <SidebarLayoutContent
      breadcrumbs={
        <Breadcrumbs>
          <BreadcrumbHome />
          <BreadcrumbSeparator />
          <Breadcrumb href="/docs/guidelines">Documentation</Breadcrumb>
          <BreadcrumbSeparator />
          <Breadcrumb>Project Guidelines</Breadcrumb>
        </Breadcrumbs>
      }
    >
      <article className="prose prose-base sm:prose-lg dark:prose-invert max-w-none lg:max-w-4xl mx-auto py-8">
      <h1>Project Guidelines</h1>

      <p>This document defines the teaching philosophy, content standards, and structural requirements for all materials in this machine learning introduction project.</p>

      <h2>Table of Contents</h2>

      <ol>
        <li><a href="#core-teaching-philosophy">Core Teaching Philosophy</a></li>
        <li><a href="#learning-science-principles">Learning Science Principles</a></li>
        <li><a href="#content-structure-requirements">Content Structure Requirements</a></li>
        <li><a href="#writing-style-guidelines">Writing Style Guidelines</a></li>
        <li><a href="#technical-accuracy-standards">Technical Accuracy Standards</a></li>
        <li><a href="#code-standards">Code Standards</a></li>
        <li><a href="#visual-and-interactive-elements">Visual and Interactive Elements</a></li>
        <li><a href="#review-checklist">Review Checklist</a></li>
      </ol>

      <hr />

      <h2 id="core-teaching-philosophy">Core Teaching Philosophy</h2>

      <h3>The "Application to Theory" Approach</h3>

      <p>Every tutorial MUST follow this sequence:</p>

      <ol>
        <li>
          <strong>Application First</strong> (Section 2)
          <ul>
            <li>Start with a real-world problem that's relatable and interesting</li>
            <li>Show working code and results BEFORE explaining theory</li>
            <li>Let learners see what the algorithm can do</li>
            <li>Build intuition through hands-on experience</li>
          </ul>
        </li>
        <li>
          <strong>Theory Second</strong> (Section 3)
          <ul>
            <li>Only after seeing it work, explain how it works</li>
            <li>Motivate every theoretical concept by connecting it back to the application</li>
            <li>Use the application as a running example throughout theory</li>
            <li>Mathematics should answer questions raised by the application</li>
          </ul>
        </li>
      </ol>

      <h3>Why This Approach?</h3>

      <p><strong>Traditional approach:</strong> "Here's 10 pages of math, now let's see why it matters"</p>
      <ul>
        <li>Result: Learners are lost, demotivated, don't see the point</li>
      </ul>

      <p><strong>Our approach:</strong> "Here's what it does (cool!), now let's understand how (motivated!)"</p>
      <ul>
        <li>Result: Learners are engaged, curious, and ready to dive deeper</li>
      </ul>

      <h3>Four Fundamental Principles</h3>

      <p>These principles apply to ALL work in this project. They are listed in order of priority:</p>

      <h4>1. Simplicity & Clarity (HIGHEST PRIORITY)</h4>
      <ul>
        <li><strong>Most important rule</strong>: If a learner doesn't understand, we have failed</li>
        <li>Prefer clear explanation over technically precise jargon</li>
        <li>Break complex concepts into digestible pieces</li>
        <li>Use analogies and real-world comparisons</li>
        <li>Progressive complexity: simple → intermediate → advanced</li>
      </ul>

      <h4>2. Absolute Correctness</h4>
      <ul>
        <li>Technical accuracy is non-negotiable</li>
        <li>All mathematical formulations must be precise</li>
        <li>All code must be tested and verified</li>
        <li>Domain/range specifications must be exact</li>
        <li>Common misconceptions must be explicitly addressed</li>
      </ul>

      <h4>3. Depth & Thoroughness</h4>
      <ul>
        <li>Never gloss over important concepts</li>
        <li>Explain the "why" behind every "what"</li>
        <li>Provide complete explanations, not summaries</li>
        <li>Include edge cases and limitations</li>
        <li>Comprehensive, not superficial</li>
      </ul>

      <h4>4. Limited Jargon</h4>
      <ul>
        <li>Minimize technical jargon whenever possible</li>
        <li>When jargon is necessary, define it immediately and clearly</li>
        <li>Use plain language as the default</li>
        <li>Technical terms should be introduced, not assumed</li>
      </ul>

      <hr />

      <h2 id="learning-science-principles">Learning Science Principles</h2>

      <p><strong>CRITICAL INSIGHT:</strong> Research in cognitive psychology shows that passive reading creates an "illusion of competence"—learners feel they understand while reading but struggle to apply knowledge independently. Our materials must engage learners in active cognitive processes that build durable, transferable understanding.</p>

      <p>This section incorporates evidence-based principles from learning science research. For complete research citations and detailed analysis, see <strong>LEARNING_SCIENCE_REVIEW.md</strong>.</p>

      <h3>Active Learning Over Passive Consumption</h3>

      <p><strong>The Problem with Passive Learning:</strong></p>
      <ul>
        <li>Reading and observing code feels easy → feels like learning</li>
        <li>But easy = weak memory formation</li>
        <li>Result: Rapid forgetting, poor transfer to new problems</li>
      </ul>

      <p><strong>The Solution:</strong></p>
      <p>Every tutorial must include active engagement opportunities. Learners should:</p>
      <ul>
        <li><strong>Retrieve</strong> information from memory (not just recognize it)</li>
        <li><strong>Generate</strong> answers before seeing solutions</li>
        <li><strong>Apply</strong> concepts to new situations</li>
        <li><strong>Reflect</strong> on their understanding</li>
      </ul>

      <h3>Five Essential Learning Science Principles</h3>

      <h4>Principle 1: Retrieval Practice (CRITICAL)</h4>

      <p><strong>Research Finding:</strong> Testing yourself is more effective than re-reading for long-term retention (Roediger & Karpicke, 2006).</p>

      <p><strong>Implementation Requirements:</strong></p>

      <p>Every tutorial MUST include:</p>

      <ol>
        <li>
          <strong>"Quick Check" sections</strong> (3-5 per tutorial)
          <ul>
            <li>After each major concept or section</li>
            <li>2-4 questions requiring recall without scrolling back</li>
            <li>Use <code>&lt;details&gt;</code> tags to hide answers</li>
            <li>Mix factual and conceptual questions</li>
          </ul>
        </li>
        <li>
          <strong>End-of-tutorial assessment</strong>
          <ul>
            <li>Comprehensive practice problems</li>
            <li>Progressive difficulty: guided → independent</li>
            <li>Both application and theory questions</li>
          </ul>
        </li>
      </ol>

      <p><strong>Example:</strong></p>
      <pre><code>{`### 🧠 Quick Check

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
</details>`}</code></pre>

      <p><strong>Why This Works:</strong> The act of retrieving information strengthens memory more than passive review. Even failed retrieval attempts improve subsequent learning.</p>

      <h4>Principle 2: Worked Examples with Faded Guidance</h4>

      <p><strong>Research Finding:</strong> Learners acquire skills most effectively through a progression from complete examples to independent problem-solving (Sweller et al., 1998).</p>

      <p><strong>Implementation Requirements:</strong></p>

      <p>For every complex skill, provide this 4-step progression:</p>

      <ol>
        <li><strong>Step 1: Fully worked example</strong> (current standard ✓)
          <ul>
            <li>Complete code with detailed explanation</li>
            <li>Show every step</li>
          </ul>
        </li>
        <li><strong>Step 2: Completion problem</strong> (MUST ADD)
          <ul>
            <li>Same problem with 20-40% blanks to fill</li>
            <li>Provide hints</li>
            <li>Hide solution in <code>&lt;details&gt;</code> tag</li>
          </ul>
        </li>
        <li><strong>Step 3: Guided problem</strong> (MUST ADD)
          <ul>
            <li>Similar problem, different context</li>
            <li>Hints available but no code scaffolding</li>
            <li>Learner writes from scratch with support</li>
          </ul>
        </li>
        <li><strong>Step 4: Independent problem</strong> (MUST ADD)
          <ul>
            <li>Novel application requiring transfer</li>
            <li>No hints, no solution provided</li>
            <li>Learner works completely independently</li>
          </ul>
        </li>
      </ol>

      <p><strong>Example Structure:</strong></p>
      <pre><code>{`### Worked Example: Linear Regression
[Complete code with full explanation]

### 🔨 Your Turn: Complete This
\`\`\`python
def gradient_descent(...):
    y_pred = _____  # Hint: mx + b
    gradient = _____  # Hint: derivative of cost
\`\`\`

### 🚀 Similar Problem
Apply to housing prices dataset (hints provided)

### 💪 Independent Challenge
Apply to your own dataset (no support)`}</code></pre>

      <p><strong>Why This Works:</strong> Gradually removing support prevents cognitive overload while building independent capability.</p>

      <h4>Principle 3: Metacognition - Think About Thinking</h4>

      <p><strong>Research Finding:</strong> Learners who monitor and regulate their understanding achieve better outcomes (Schraw & Dennison, 1994).</p>

      <p><strong>Implementation Requirements:</strong></p>

      <p>Every tutorial MUST include:</p>

      <ol>
        <li><strong>Learning objectives before complex sections</strong>
          <pre><code>{`### 📍 Before We Begin: Gradient Descent

After this section, you should be able to:
- Explain why we need optimization
- Calculate one gradient descent step
- Identify when it might fail`}</code></pre>
        </li>
        <li><strong>Reflection checkpoints after complex sections</strong>
          <pre><code>{`### 🤔 Reflection

Take 2 minutes:
1. What was the main insight?
2. What's still confusing?
3. Can you explain this to a friend?

Rate your understanding (1-5): [ ]
If below 3, review the section`}</code></pre>
        </li>
        <li><strong>Self-assessment opportunities</strong>
          <ul>
            <li>Confidence ratings</li>
            <li>Self-explanation prompts</li>
            <li>"What questions do you still have?"</li>
          </ul>
        </li>
      </ol>

      <p><strong>Why This Works:</strong> Metacognitive awareness helps learners identify gaps, allocate study time effectively, and develop self-directed learning skills.</p>

      <h4>Principle 4: Address Misconceptions Explicitly</h4>

      <p><strong>Research Finding:</strong> Misconceptions persist unless directly confronted and replaced with correct understanding.</p>

      <p><strong>Implementation Requirements:</strong></p>

      <p>Every tutorial MUST include 3-5 "Common Misconceptions" boxes:</p>

      <pre><code>{`### ⚠️ Common Misconceptions

**Misconception 1: "R² always ranges from 0 to 1"**
- ❌ **Why it's wrong:** R² can be negative!
- ✅ **Correct:** R² ranges from -∞ to 1. Negative means worse than baseline.
- 🧠 **Check:** If R²=-0.5, what does that mean?
  <details><summary>Answer</summary>
  Model performs worse than just predicting the mean
  </details>`}</code></pre>

      <p><strong>Required Elements:</strong></p>
      <ul>
        <li>State the misconception explicitly</li>
        <li>Explain why it's tempting/wrong</li>
        <li>Provide correct understanding</li>
        <li>Give concrete example showing the difference</li>
        <li>Include self-check question</li>
      </ul>

      <p><strong>Why This Works:</strong> Merely presenting correct information doesn't erase misconceptions. They must be explicitly identified and refuted.</p>

      <h4>Principle 5: Spaced Repetition and Interleaving</h4>

      <p><strong>Research Finding:</strong> Distributing practice over time with mixing of topics improves long-term retention by 20-30% (Cepeda et al., 2006).</p>

      <p><strong>Implementation Requirements:</strong></p>

      <ol>
        <li><strong>Within tutorials:</strong> Revisit concepts
          <ul>
            <li>Reference earlier concepts when introducing new ones</li>
            <li>"Recall from Linear Regression that we split data..."</li>
            <li>Build cumulative understanding</li>
          </ul>
        </li>
        <li><strong>Across tutorials:</strong> Interleave content
          <ul>
            <li>KNN tutorial should review train/test split from Linear Regression</li>
            <li>Later tutorials mix old and new concepts</li>
            <li>Create cumulative review notebooks</li>
          </ul>
        </li>
        <li><strong>Spaced retrieval schedules:</strong>
          <pre><code>{`### 🔄 Spaced Review

To maximize retention:
- ✅ Today: Complete all challenges
- 📅 Tomorrow: 5-minute quiz
- 📅 Next week: Mixed practice
- 📅 Next month: Cumulative review`}</code></pre>
        </li>
      </ol>

      <p><strong>Why This Works:</strong> Spacing creates "desirable difficulties" that strengthen memory. Interleaving improves discrimination between concepts and transfer.</p>

      <h3>Additional Evidence-Based Techniques</h3>

      <h4>Prediction Prompts (Generation Effect)</h4>

      <p>Before revealing answers, ask learners to predict:</p>

      <pre><code>{`### 🔮 Before Running Code

Predict:
1. Will training or test accuracy be higher?
2. What R² range do you expect?

**Now run and compare to your prediction**

Reflection:
- Were you close?
- What surprised you?`}</code></pre>

      <p><strong>Why This Works:</strong> Generating an answer (even wrong) before seeing the correct one improves learning more than passive reading.</p>

      <h4>Concrete-Representational-Abstract (CRA) Sequence</h4>

      <p>For complex concepts, use this progression:</p>

      <ol>
        <li><strong>Concrete:</strong> Physical analogy or real-world example</li>
        <li><strong>Representational:</strong> Visual diagram or animation</li>
        <li><strong>Abstract:</strong> Mathematical formula or general principle</li>
      </ol>

      <p><strong>Example: Gradient Descent</strong></p>
      <ul>
        <li>Concrete: "Blindfolded on a hill, feeling for downslope"</li>
        <li>Representational: Animated visualization of ball rolling down surface</li>
        <li>Abstract: $m := m - \alpha \frac{'{'}∂J{'}'}{'{'}∂m{'}'}$</li>
      </ul>

      <h4>Elaborative Interrogation ("Why" Questions)</h4>

      <p>Prompt learners to explain why things work:</p>

      <pre><code>{`### 🤔 Deep Question

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

</details>`}</code></pre>

      <h3>Balancing Cognitive Load</h3>

      <p><strong>Working Memory Limitations:</strong></p>
      <ul>
        <li>Humans can hold ~4 chunks of information in working memory</li>
        <li>Excessive load prevents learning</li>
      </ul>

      <p><strong>Guidelines:</strong></p>

      <ol>
        <li><strong>Chunk information meaningfully</strong>
          <ul>
            <li>Group related concepts</li>
            <li>Use the "What, Why, How" pattern</li>
            <li>Break complex derivations into steps</li>
          </ul>
        </li>
        <li><strong>Reduce extraneous load</strong>
          <ul>
            <li>Clear formatting and organization</li>
            <li>Minimize distracting elements</li>
            <li>Consistent structure across tutorials</li>
          </ul>
        </li>
        <li><strong>Manage intrinsic load</strong>
          <ul>
            <li>Start simple, build complexity gradually</li>
            <li>Use the application-to-theory approach</li>
            <li>Provide prerequisites clearly</li>
          </ul>
        </li>
        <li><strong>Optimize germane load</strong>
          <ul>
            <li>Focus attention on schema building</li>
            <li>Connect new to prior knowledge</li>
            <li>Make patterns explicit</li>
          </ul>
        </li>
      </ol>

      <h3>Motivation and Self-Efficacy</h3>

      <p><strong>Research Finding:</strong> Self-efficacy (belief in one's ability) predicts learning outcomes better than past achievement (Bandura).</p>

      <p><strong>Implementation Strategies:</strong></p>

      <ol>
        <li><strong>Normalize struggle</strong>
          <pre><code>{`### 💪 Feeling Confused?

That's completely normal! Gradient descent is challenging.
Even ML experts found this hard at first.

Struggle is a sign your brain is growing.`}</code></pre>
        </li>
        <li><strong>Provide early wins</strong>
          <ul>
            <li>Start with achievable challenges</li>
            <li>Celebrate progress</li>
            <li>Build confidence progressively</li>
          </ul>
        </li>
        <li><strong>Growth mindset messaging</strong>
          <ul>
            <li>Emphasize effort over innate ability</li>
            <li>Frame mistakes as learning opportunities</li>
            <li>Provide specific strategies when stuck</li>
          </ul>
        </li>
        <li><strong>Real-world relevance</strong>
          <ul>
            <li>Show why concepts matter</li>
            <li>Connect to career applications</li>
            <li>Highlight practical impact</li>
          </ul>
        </li>
      </ol>

      <h3>Quality Checklist for Learning Science Integration</h3>

      <p>Before releasing any tutorial, verify:</p>

      <p><strong>Active Learning:</strong></p>
      <ul>
        <li>[ ] 3-5 "Quick Check" retrieval sections</li>
        <li>[ ] End-of-tutorial practice challenges</li>
        <li>[ ] Prediction prompts before reveals</li>
      </ul>

      <p><strong>Worked Examples:</strong></p>
      <ul>
        <li>[ ] Complete example → Completion → Guided → Independent progression</li>
        <li>[ ] Faded guidance for complex skills</li>
        <li>[ ] Multiple practice opportunities</li>
      </ul>

      <p><strong>Metacognition:</strong></p>
      <ul>
        <li>[ ] Learning objectives before complex sections</li>
        <li>[ ] Reflection checkpoints after complex sections</li>
        <li>[ ] Self-assessment throughout</li>
      </ul>

      <p><strong>Misconceptions:</strong></p>
      <ul>
        <li>[ ] 3-5 common misconceptions explicitly addressed</li>
        <li>[ ] Explain why misconception is tempting</li>
        <li>[ ] Provide concrete corrective examples</li>
      </ul>

      <p><strong>Spacing & Interleaving:</strong></p>
      <ul>
        <li>[ ] Concepts revisited within tutorial</li>
        <li>[ ] References to previous tutorials</li>
        <li>[ ] Spaced review schedule provided</li>
      </ul>

      <p><strong>Motivation:</strong></p>
      <ul>
        <li>[ ] Struggle normalized with support strategies</li>
        <li>[ ] Early achievable challenges</li>
        <li>[ ] Growth mindset messaging</li>
        <li>[ ] Real-world relevance emphasized</li>
      </ul>

      <h3>Implementation Priority</h3>

      <p><strong>Phase 1: Critical (Do First)</strong></p>
      <ol>
        <li>Add retrieval practice (highest impact per hour invested)</li>
        <li>Address misconceptions explicitly</li>
        <li>Add end-of-tutorial challenges</li>
      </ol>

      <p><strong>Phase 2: High Value (Do Second)</strong></p>
      <ol start={4}>
        <li>Add learning objectives and reflection checkpoints</li>
        <li>Implement worked example progressions</li>
        <li>Add prediction prompts</li>
      </ol>

      <p><strong>Phase 3: Enhancing (Do Third)</strong></p>
      <ol start={7}>
        <li>Build spaced repetition system</li>
        <li>Add multiple contexts for transfer</li>
        <li>Implement CRA sequences for complex topics</li>
      </ol>

      <h3>Measuring Effectiveness</h3>

      <p>Track these metrics to verify improvements:</p>

      <p><strong>Learning Outcomes:</strong></p>
      <ul>
        <li>Quiz performance (immediate and delayed)</li>
        <li>Transfer task success rate</li>
        <li>Completion rates</li>
        <li>Time to competency</li>
      </ul>

      <p><strong>Learner Experience:</strong></p>
      <ul>
        <li>Self-reported understanding</li>
        <li>Confidence ratings</li>
        <li>Satisfaction scores</li>
      </ul>

      <p><strong>Expected Improvements with Full Implementation:</strong></p>
      <ul>
        <li>+20-30% on delayed retention tests</li>
        <li>+15-25% on transfer tasks</li>
        <li>+10-15% completion rates</li>
        <li>+25-35% self-reported confidence</li>
      </ul>

      <hr />

      <h2 id="content-structure-requirements">Content Structure Requirements</h2>

      <h3>Mandatory Structure</h3>

      <p>Every algorithm tutorial MUST follow this exact structure:</p>

      <pre><code>{`# [Algorithm Name]: From Application to Theory

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
- Links to related algorithms`}</code></pre>

      <h3>Section Numbering</h3>

      <ul>
        <li><strong>Section 1</strong>: Introduction</li>
        <li><strong>Section 2</strong>: Application (practical, hands-on)</li>
        <li><strong>Section 3</strong>: Theory (mathematical, conceptual)</li>
        <li><strong>Section 4</strong>: Conclusion and next steps</li>
      </ul>

      <p>This numbering is MANDATORY and consistent across all tutorials.</p>

      <hr />

      <h2 id="writing-style-guidelines">Writing Style Guidelines</h2>

      <h3>Tone and Voice</h3>

      <ul>
        <li><strong>Conversational but professional</strong>
          <ul>
            <li>✅ "Let's see what happens when we run this code"</li>
            <li>❌ "The subsequent execution of the aforementioned code block"</li>
          </ul>
        </li>
        <li><strong>Encouraging and inclusive</strong>
          <ul>
            <li>✅ "This might seem complex at first, but let's break it down"</li>
            <li>❌ "Obviously, this is straightforward"</li>
          </ul>
        </li>
        <li><strong>Direct and active</strong>
          <ul>
            <li>✅ "We split the data into training and testing sets"</li>
            <li>❌ "The data is split into training and testing sets"</li>
          </ul>
        </li>
      </ul>

      <h3>Explaining Concepts</h3>

      <h4>The "What, Why, How" Pattern</h4>

      <p>For every new concept, answer in this order:</p>

      <ol>
        <li><strong>What</strong> - Define it in plain language</li>
        <li><strong>Why</strong> - Why do we need this? What problem does it solve?</li>
        <li><strong>How</strong> - How does it work? What's the implementation?</li>
      </ol>

      <p>Example:</p>
      <pre><code>{`### Train/Test Split

**What**: Dividing your dataset into two parts - one for training, one for testing.

**Why**: We need to test our model on data it has never seen before to ensure it's
actually learning patterns, not just memorizing the training data.

**How**: We use sklearn's train_test_split function, which randomly shuffles and
divides the data according to the ratio we specify.`}</code></pre>

      <h4>Introducing Mathematics</h4>

      <ul>
        <li><strong>Always motivate before formulating</strong>
          <ul>
            <li>❌ "The cost function is: J(θ) = ..."</li>
            <li>✅ "We need a way to measure how wrong our predictions are. This measurement is called a cost function: J(θ) = ..."</li>
          </ul>
        </li>
        <li><strong>Explain every symbol</strong>
          <pre><code>{`The equation is:

$$ y = mx + b $$

Where:
- $y$ is the predicted value
- $m$ is the slope of the line
- $x$ is the input feature
- $b$ is the y-intercept`}</code></pre>
        </li>
        <li><strong>Provide intuition alongside formulas</strong>
          <pre><code>{`$$ MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 $$

In plain English: we take each prediction error, square it (to make negatives positive
and to penalize large errors more), then average all these squared errors.`}</code></pre>
        </li>
      </ul>

      <h3>Avoiding Common Pitfalls</h3>

      <p>❌ <strong>Don't assume knowledge</strong></p>
      <ul>
        <li>Bad: "Using standard gradient descent"</li>
        <li>Good: "Using gradient descent, an optimization algorithm that we'll explore in detail in Section 3.6"</li>
      </ul>

      <p>❌ <strong>Don't skip steps</strong></p>
      <ul>
        <li>Bad: "Therefore, the optimal parameters are..."</li>
        <li>Good: "Let's work through this step by step: First, we... Then we... Therefore..."</li>
      </ul>

      <p>❌ <strong>Don't use undefined acronyms</strong></p>
      <ul>
        <li>Bad: "Use MSE for regression tasks"</li>
        <li>Good: "Use MSE (Mean Squared Error) for regression tasks"</li>
      </ul>

      <p>✅ <strong>Do provide context</strong></p>
      <ul>
        <li>"In our stock price prediction example, this means..."</li>
        <li>"Going back to the Iris dataset we loaded earlier..."</li>
      </ul>

      <p>✅ <strong>Do acknowledge complexity</strong></p>
      <ul>
        <li>"This is the most mathematically intensive part, so take your time"</li>
        <li>"If this seems confusing, that's normal - we'll clarify with an example"</li>
      </ul>

      <hr />

      <h2 id="technical-accuracy-standards">Technical Accuracy Standards</h2>

      <h3>Mathematical Precision</h3>

      <ol>
        <li><strong>Domain and Range</strong>
          <ul>
            <li>Always specify domain and range for functions</li>
            <li>Use correct interval notation: (0, 1) vs [0, 1]</li>
            <li>Example: "The sigmoid function has domain (-∞, ∞) and range (0, 1)"</li>
          </ul>
        </li>
        <li><strong>Asymptotic Behavior</strong>
          <ul>
            <li>Clarify when functions approach but never reach values</li>
            <li>Example: "The sigmoid approaches 0 and 1 asymptotically but never actually reaches these values"</li>
          </ul>
        </li>
        <li><strong>Complexity Notation</strong>
          <ul>
            <li>Use Big-O notation correctly</li>
            <li>Specify what variables represent: "O(N·d) where N is samples and d is dimensions"</li>
          </ul>
        </li>
        <li><strong>Statistical Metrics</strong>
          <ul>
            <li>Specify bounds correctly: "R² ranges from -∞ to 1, not 0 to 1"</li>
            <li>Explain special cases: "R² can be negative when the model performs worse than predicting the mean"</li>
          </ul>
        </li>
      </ol>

      <h3>Common Technical Mistakes to Avoid</h3>

      <p>❌ <strong>Imprecise range descriptions</strong></p>
      <ul>
        <li>Bad: "Sigmoid outputs values between 0 and 1"</li>
        <li>Good: "Sigmoid outputs values in the open interval (0, 1), approaching but never reaching 0 or 1"</li>
      </ul>

      <p>❌ <strong>Incorrect complexity</strong></p>
      <ul>
        <li>Bad: "KNN has O(N²) complexity"</li>
        <li>Good: "Brute-force KNN has O(N·d) complexity for a single prediction, where N is training samples and d is dimensions"</li>
      </ul>

      <p>❌ <strong>Ambiguous array indexing</strong></p>
      <ul>
        <li>Bad: <code>cm[0][0] # True Negative</code></li>
        <li>Good: <code>cm[0][0] # True Negative (assuming class 0 is negative - sklearn sorts labels)</code></li>
      </ul>

      <h3>Code Accuracy</h3>

      <ol>
        <li><strong>Reproducibility</strong>
          <ul>
            <li>Always set <code>random_state</code> in stochastic operations</li>
            <li>Example: <code>train_test_split(X, y, test_size=0.2, random_state=42)</code></li>
          </ul>
        </li>
        <li><strong>Correct Function Usage</strong>
          <ul>
            <li>Verify function signatures match actual usage</li>
            <li>Include all necessary parameters</li>
            <li>Document assumptions</li>
          </ul>
        </li>
        <li><strong>Error Handling</strong>
          <ul>
            <li>Code should handle edge cases appropriately</li>
            <li>Include comments for non-obvious behavior</li>
          </ul>
        </li>
      </ol>

      <hr />

      <h2 id="code-standards">Code Standards</h2>

      <h3>Code Quality</h3>

      <ol>
        <li><strong>Readability</strong>
          <pre><code>{`# ✅ GOOD - Clear variable names, commented
# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ❌ BAD - Unclear, no context
a, b, c, d = train_test_split(X, y, test_size=0.2)`}</code></pre>
        </li>
        <li><strong>Comments</strong>
          <ul>
            <li>Explain WHY, not WHAT</li>
            <li>✅ "# Use log scale because feature values span many orders of magnitude"</li>
            <li>❌ "# Convert to log"</li>
          </ul>
        </li>
        <li><strong>Structure</strong>
          <ul>
            <li>Imports at the top of each section where first needed</li>
            <li>Explain new imports when introduced</li>
            <li>Logical grouping of related code</li>
          </ul>
        </li>
      </ol>

      <h3>Code Explanations</h3>

      <p>Every code cell should be:</p>
      <ol>
        <li><strong>Preceded by explanation</strong> - What we're about to do and why</li>
        <li><strong>Followed by interpretation</strong> - What the output means</li>
      </ol>

      <p>Example:</p>
      <pre><code>{`Now let's train our model on the training data. Training means the model will
learn the patterns in the data by adjusting its internal parameters.

[CODE CELL]

The model has now learned from 80% of our data. Next, we'll test it on the
remaining 20% that it hasn't seen before.`}</code></pre>

      <hr />

      <h2 id="visual-and-interactive-elements">Visual and Interactive Elements</h2>

      <h3>Required Visualizations</h3>

      <ol>
        <li><strong>Data Exploration</strong>
          <ul>
            <li>Always visualize the data before modeling</li>
            <li>Use scatter plots, histograms, or appropriate visualization</li>
            <li>Label axes clearly</li>
            <li>Include titles that explain what you're showing</li>
          </ul>
        </li>
        <li><strong>Results Visualization</strong>
          <ul>
            <li>Show predictions vs actual values</li>
            <li>Include decision boundaries for classification</li>
            <li>Use color coding effectively</li>
            <li>Add legends and labels</li>
          </ul>
        </li>
        <li><strong>Performance Metrics</strong>
          <ul>
            <li>Visualize training curves when relevant</li>
            <li>Show confusion matrices for classification</li>
            <li>Use appropriate scale (linear vs log)</li>
          </ul>
        </li>
      </ol>

      <h3>Visualization Standards</h3>

      <pre><code>{`# ✅ GOOD - Complete labeling and context
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
plt.show()`}</code></pre>

      <h3>Interactive Elements</h3>

      <p>Where possible, include:</p>
      <ul>
        <li>Multiple examples with different parameters</li>
        <li>Comparison of different approaches</li>
        <li>What-if scenarios</li>
        <li>Failure cases and limitations</li>
      </ul>

      <hr />

      <h2 id="review-checklist">Review Checklist</h2>

      <p>Before submitting any content, verify ALL of the following:</p>

      <h3>Structure Review</h3>
      <ul>
        <li>[ ] Follows "Application to Theory" structure</li>
        <li>[ ] Section 1: Introduction is clear and motivating</li>
        <li>[ ] Section 2: Application is complete and working</li>
        <li>[ ] Section 3: Theory is thorough and well-motivated</li>
        <li>[ ] Table of contents is present and accurate</li>
        <li>[ ] All section numbers are consistent</li>
      </ul>

      <h3>Content Review</h3>
      <ul>
        <li>[ ] Every technical term is defined when first introduced</li>
        <li>[ ] Every mathematical formula has an intuitive explanation</li>
        <li>[ ] Every symbol in equations is explained</li>
        <li>[ ] Code examples work correctly</li>
        <li>[ ] All visualizations have proper labels and titles</li>
        <li>[ ] Links to other sections/notebooks work correctly</li>
      </ul>

      <h3>Technical Accuracy Review</h3>
      <ul>
        <li>[ ] All mathematical statements are precise</li>
        <li>[ ] Domain and range specifications are correct</li>
        <li>[ ] Complexity analysis is accurate</li>
        <li>[ ] Statistical metrics are correctly bounded</li>
        <li>[ ] Code includes <code>random_state</code> where needed</li>
        <li>[ ] Array indexing assumptions are documented</li>
        <li>[ ] No common misconceptions are present</li>
      </ul>

      <h3>Style Review</h3>
      <ul>
        <li>[ ] Writing is clear and accessible</li>
        <li>[ ] Tone is conversational but professional</li>
        <li>[ ] Jargon is minimized and explained</li>
        <li>[ ] "What, Why, How" pattern is followed</li>
        <li>[ ] Active voice is used</li>
        <li>[ ] Concepts build progressively</li>
      </ul>

      <h3>Code Review</h3>
      <ul>
        <li>[ ] Variable names are descriptive</li>
        <li>[ ] Comments explain WHY, not WHAT</li>
        <li>[ ] Code is preceded by explanation</li>
        <li>[ ] Output is interpreted and explained</li>
        <li>[ ] Edge cases are handled or documented</li>
        <li>[ ] Imports are explained when first introduced</li>
      </ul>

      <h3>Completeness Review</h3>
      <ul>
        <li>[ ] No "TODO" or placeholder sections</li>
        <li>[ ] All referenced sections exist</li>
        <li>[ ] All figures and images display correctly</li>
        <li>[ ] Examples are complete and tested</li>
        <li>[ ] Conclusion summarizes key points</li>
      </ul>

      <hr />

      <h2>Examples of Good vs Bad Content</h2>

      <h3>Example 1: Introducing a Concept</h3>

      <p>❌ <strong>BAD</strong></p>
      <pre><code>{`### Mean Squared Error

$$ MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 $$

This is the cost function we minimize.`}</code></pre>

      <p>✅ <strong>GOOD</strong></p>
      <pre><code>{`### Mean Squared Error (MSE)

We need a way to measure how wrong our predictions are. One popular approach is
Mean Squared Error (MSE).

**What it does**: MSE calculates the average of all squared prediction errors.

**Why square the errors**:
- Makes negative errors positive (a prediction that's 5 too high is just as
  bad as 5 too low)
- Penalizes large errors more than small ones (being off by 10 is more than
  twice as bad as being off by 5)

**The Formula**:

$$ MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 $$

Where:
- $n$ is the number of predictions
- $y_i$ is the actual value for example $i$
- $\\hat{y}_i$ is our predicted value for example $i$
- $(y_i - \\hat{y}_i)$ is the prediction error

**In plain English**: For each prediction, calculate how far off you were, square
that error, then average all the squared errors together.

In our stock price example, if we predicted $100 but the actual price was $105,
the error would be -5, and the squared error would be 25.`}</code></pre>

      <h3>Example 2: Code Introduction</h3>

      <p>❌ <strong>BAD</strong></p>
      <pre><code>{`from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`}</code></pre>

      <p>✅ <strong>GOOD</strong></p>
      <pre><code>{`Now that we have our features (X) and targets (y), we need to split them into
training and testing sets. This is a crucial step in machine learning.

**Why split the data?**
We want to train our model on one portion of the data and test it on data it
has never seen before. This tells us if the model is actually learning patterns
or just memorizing the training data.

We'll use sklearn's \`train_test_split\` function, which randomly divides our
data. We'll use 80% for training and 20% for testing.

[CODE CELL]
\`\`\`python
from sklearn.model_selection import train_test_split

# Split: 80% training, 20% testing
# random_state=42 ensures we get the same split every time (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
\`\`\`

Now we have:
- \`X_train\` and \`y_train\`: 80% of the data for training
- \`X_test\` and \`y_test\`: 20% of the data for testing (model has never seen this)`}</code></pre>

      <hr />

      <h2>Maintaining These Standards</h2>

      <h3>For New Content</h3>

      <ol>
        <li>Review this document before starting</li>
        <li>Use existing tutorials (Linear Regression, KNN) as templates</li>
        <li>Follow the checklist before submitting</li>
        <li>Ask: "Would I understand this if I knew nothing about ML?"</li>
      </ol>

      <h3>For Content Updates</h3>

      <ol>
        <li>Ensure changes maintain the application-to-theory structure</li>
        <li>Don't sacrifice clarity for brevity</li>
        <li>Don't sacrifice accuracy for simplicity</li>
        <li>Find the balance: accurate AND understandable</li>
      </ol>

      <h3>Priority Order When Conflicts Arise</h3>

      <p>If you must choose between competing concerns, prioritize in this order:</p>

      <ol>
        <li><strong>Correctness</strong> - Never sacrifice technical accuracy</li>
        <li><strong>Clarity</strong> - If correct but unclear, simplify the explanation (not the accuracy)</li>
        <li><strong>Completeness</strong> - Better to be thorough than brief</li>
        <li><strong>Consistency</strong> - Follow established patterns</li>
      </ol>

      <hr />

      <h2>Conclusion</h2>

      <p>These guidelines exist to ensure every learner, regardless of background, can understand and apply machine learning concepts. When in doubt, ask:</p>

      <ul>
        <li>"Would this make sense to someone learning ML for the first time?"</li>
        <li>"Am I showing them why this matters before diving into how it works?"</li>
        <li>"Is every statement technically accurate?"</li>
        <li>"Have I explained every piece of jargon?"</li>
      </ul>

      <p>Remember: <strong>If the learner doesn't understand, we haven't taught effectively.</strong></p>

      <p>The goal is not to impress with complexity, but to illuminate with clarity.</p>
      </article>
    </SidebarLayoutContent>
  );
}
