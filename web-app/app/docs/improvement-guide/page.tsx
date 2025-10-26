import type { Metadata } from 'next';
import { SidebarLayoutContent } from '@/components/layout/sidebar-layout';
import {
  Breadcrumbs,
  BreadcrumbHome,
  BreadcrumbSeparator,
  Breadcrumb,
} from '@/components/layout/breadcrumbs';

export const metadata: Metadata = {
  title: "Improvement Guide",
  description: "Practical templates and examples for implementing active learning principles in ML tutorials.",
  openGraph: {
    title: "Improvement Guide | ML Introduction",
    description: "Practical templates and examples for implementing active learning principles in ML tutorials.",
    type: "article",
  },
};

export default function ImprovementGuidePage() {
  return (
    <SidebarLayoutContent
      breadcrumbs={
        <Breadcrumbs>
          <BreadcrumbHome />
          <BreadcrumbSeparator />
          <Breadcrumb href="/docs/improvement-guide">Documentation</Breadcrumb>
          <BreadcrumbSeparator />
          <Breadcrumb>Improvement Guide</Breadcrumb>
        </Breadcrumbs>
      }
    >
      <article className="prose prose-base sm:prose-lg dark:prose-invert max-w-none lg:max-w-4xl mx-auto py-8">
      <h1>Machine Learning Tutorial Improvement Guide</h1>

      <p><strong>Quick Reference for Evidence-Based Enhancements</strong></p>

      <p>This guide provides practical templates and strategies for transforming passive tutorials into active learning experiences. All recommendations are based on cognitive science research (see LEARNING_SCIENCE_REVIEW.md for full citations).</p>

      <hr />

      <h2>The Core Problem</h2>

      <p><strong>Current State:</strong> Learners passively read and observe → Illusion of competence → Poor retention</p>
      <p><strong>Goal State:</strong> Learners actively engage and retrieve → Deep understanding → Long-term mastery</p>

      <p><strong>Research shows:</strong> Retrieval practice, metacognition, and worked examples are among the most powerful learning techniques.</p>

      <hr />

      <h2>Quick Implementation Checklist</h2>

      <h3>For Every Tutorial Section:</h3>

      <ul>
        <li>[ ] <strong>Before new concepts:</strong> Add learning objectives and pre-assessment</li>
        <li>[ ] <strong>During explanations:</strong> Insert prediction prompts</li>
        <li>[ ] <strong>After code examples:</strong> Add "Quick Check" retrieval questions</li>
        <li>[ ] <strong>For complex topics:</strong> Include completion problems (fill-in-the-blank)</li>
        <li>[ ] <strong>Address 3-5 common misconceptions explicitly</strong></li>
        <li>[ ] <strong>End each major section:</strong> Reflection checkpoint</li>
        <li>[ ] <strong>End of tutorial:</strong> Comprehensive practice challenges</li>
      </ul>

      <hr />

      <h2>Ready-to-Use Templates</h2>

      <h3>Template 1: Learning Objectives (Before Complex Sections)</h3>

      <pre><code>{`### 📍 Before We Begin: [Topic Name]

[Optional: Set expectations about difficulty]
This is one of the more challenging concepts. Take your time!

**After this section, you should be able to:**
- [Specific skill 1]
- [Specific skill 2]
- [Specific skill 3]

**Self-Check:**
- Have you completed the prerequisites? [Link]
- Rate your current understanding: 😰 Confused | 😐 Uncertain | 😊 Confident`}</code></pre>

      <p><strong>Example:</strong></p>
      <pre><code>{`### 📍 Before We Begin: Gradient Descent

This is one of the most important—and challenging—concepts in machine learning.

**After this section, you should be able to:**
- Explain why we need an optimization algorithm
- Describe the role of the learning rate
- Calculate one step of gradient descent by hand

**Self-Check:**
- Are you comfortable with derivatives? (If not, see Calculus Primer)
- Rate your confidence: 😰 😐 😊`}</code></pre>

      <hr />

      <h3>Template 2: Prediction Prompts (Before Running Code)</h3>

      <pre><code>{`### 🔮 Prediction Challenge

Before running this code, predict what you'll see:

**Questions:**
1. [Prediction question 1]
2. [Prediction question 2]
3. [Why question - asks for reasoning]

<details>
<summary>Think about these hints</summary>

[Optional hints to guide thinking]

</details>

**Now run the code and see if you were right!**

[CODE CELL]

**Reflection:**
- Were your predictions correct?
- What surprised you?
- What does this tell you about [concept]?`}</code></pre>

      <p><strong>Example:</strong></p>
      <pre><code>{`### 🔮 Prediction Challenge

Before running this linear regression code, predict:

**Questions:**
1. Will the R² be higher on training data or test data? Why?
2. Will the R² be closer to 0.5, 0.8, or 0.95?
3. Why can't R² ever equal exactly 1.0 with real-world data?

**Now run the code!**

\`\`\`python
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R²: {train_score:.3f}")
print(f"Test R²: {test_score:.3f}")
\`\`\`

**Reflection:**
- Were you close?
- If train R² >> test R², what does that suggest?`}</code></pre>

      <hr />

      <h3>Template 3: Quick Check (Retrieval Practice After Explanations)</h3>

      <pre><code>{`### 🧠 Quick Check: Test Your Understanding

Try to answer without scrolling back:

**Question 1:** [Factual recall question]
<details>
<summary>Answer</summary>

[Answer with brief explanation]

</details>

**Question 2:** [Conceptual question]
<details>
<summary>Answer</summary>

[Answer with explanation]

</details>

**Question 3:** [Application question]
<details>
<summary>Answer</summary>

[Answer with example]

</details>`}</code></pre>

      <p><strong>Example:</strong></p>
      <pre><code>{`### 🧠 Quick Check: Train/Test Split

Try to answer without scrolling back:

**1. What are the two required inputs for train_test_split()?**
<details>
<summary>Answer</summary>

Features (X) and targets (y). The function needs to know what to split!

</details>

**2. Why do we split data instead of using everything for training?**
<details>
<summary>Answer</summary>

To evaluate performance on unseen data and detect overfitting. If we test on training data, we can't tell if the model learned patterns or just memorized.

</details>

**3. What would happen if we accidentally used X_test to train the model?**
<details>
<summary>Answer</summary>

We'd have no independent data to evaluate with. Our performance metrics would be overoptimistic because we'd be testing on data the model has already seen.

</details>`}</code></pre>

      <hr />

      <h3>Template 4: Completion Problems (After Worked Examples)</h3>

      <pre><code>{`### 🔨 Your Turn: Complete This Code

We just saw a complete example. Now try filling in the blanks:

\`\`\`python
# [Brief description of what this code does]

def function_name(parameters):
    """Docstring"""
    # Step 1: [Description]
    variable = _____  # Hint: [helpful hint]

    # Step 2: [Description]
    result = _____ # Hint: [helpful hint]

    return result
\`\`\`

**Hints:**
- [Hint 1]
- [Hint 2]

<details>
<summary>Show solution</summary>

\`\`\`python
[Complete solution with comments explaining key parts]
\`\`\`

**Explanation:**
[Why this solution works]

</details>`}</code></pre>

      <p><strong>Example:</strong></p>
      <pre><code>{`### 🔨 Your Turn: Complete Gradient Descent

\`\`\`python
def gradient_descent_step(x, y, m, b, learning_rate=0.01):
    """Perform one iteration of gradient descent"""
    n = len(x)

    # Step 1: Calculate predictions
    y_pred = _____  # Hint: Use m, x, and b (y = mx + b)

    # Step 2: Calculate error
    error = _____  # Hint: prediction minus actual

    # Step 3: Calculate gradients
    gradient_m = (2 / n) * np.sum(_____ * x)  # Hint: error times x
    gradient_b = (2 / n) * np.sum(_____)      # Hint: just the error

    # Step 4: Update parameters
    m_new = m - _____ * gradient_m  # Hint: learning_rate
    b_new = b - _____ * gradient_b

    return m_new, b_new
\`\`\`

<details>
<summary>Show solution</summary>

\`\`\`python
def gradient_descent_step(x, y, m, b, learning_rate=0.01):
    n = len(x)
    y_pred = m * x + b
    error = y_pred - y
    gradient_m = (2 / n) * np.sum(error * x)
    gradient_b = (2 / n) * np.sum(error)
    m_new = m - learning_rate * gradient_m
    b_new = b - learning_rate * gradient_b
    return m_new, b_new
\`\`\`

This implements the core gradient descent update rule!

</details>`}</code></pre>

      <hr />

      <h3>Template 5: Common Misconceptions</h3>

      <pre><code>{`### ⚠️ Common Misconceptions

**Misconception 1: "[Common incorrect belief]"**
- ❌ **Why it's wrong:** [Explanation]
- ✅ **Correct understanding:** [Accurate explanation]
- 🔍 **Example:** [Concrete example showing the difference]
- 🧠 **Check yourself:** [Question to verify understanding]

**Misconception 2: "[Another misconception]"**
[Same structure as above]`}</code></pre>

      <p><strong>Example:</strong></p>
      <pre><code>{`### ⚠️ Common Misconceptions About R²

**Misconception 1: "R² always ranges from 0 to 1"**
- ❌ **Why it's wrong:** R² can actually be negative!
- ✅ **Correct understanding:** R² ranges from -∞ to 1. Negative values mean your model performs worse than just predicting the mean every time.
- 🔍 **Example:** If you predict random numbers, R² could be -5.0
- 🧠 **Check yourself:** If R² = -0.5, what does that mean?
  <details><summary>Answer</summary>
  The model is terrible—worse than the simplest possible baseline
  </details>

**Misconception 2: "Higher R² is always better"**
- ❌ **Why it's wrong:** High training R² with low test R² indicates overfitting
- ✅ **Correct understanding:** Look at both training and test R². They should be similar.
- 🔍 **Example:** Train R²=0.99, Test R²=0.40 → Overfitting problem!
- 🧠 **Check yourself:** Which is better: (A) Train=0.85, Test=0.82 or (B) Train=0.95, Test=0.45?
  <details><summary>Answer</summary>
  (A) is better—similar scores indicate good generalization
  </details>`}</code></pre>

      <hr />

      <h3>Template 6: Reflection Checkpoint (After Complex Sections)</h3>

      <pre><code>{`### 🤔 Reflection Checkpoint

Take 2-3 minutes to reflect:

**Understanding Check:**
1. What was the main insight you gained from this section?
2. What's still confusing or unclear?
3. How would you explain [concept] to a friend?

**Application:**
4. Where might you use this in a real project?
5. What questions do you still have?

**Confidence Rating:**
Rate your understanding: 1 (very confused) to 5 (could teach it)
- [ ] 1 - Need to review
- [ ] 2 - Partially understand
- [ ] 3 - Understand basics
- [ ] 4 - Understand well
- [ ] 5 - Could explain to others

If you rated below 3, consider:
- Re-reading the section
- Trying the examples yourself
- Discussing with peers
- Reviewing prerequisites`}</code></pre>

      <hr />

      <h3>Template 7: End-of-Tutorial Practice Challenges</h3>

      <pre><code>{`## 5. Test Your Mastery

### 🎯 Self-Assessment

Rate your confidence (1-5) in each objective:

**Application Skills:**
- [ ] [Skill 1]
- [ ] [Skill 2]
- [ ] [Skill 3]

**Theoretical Understanding:**
- [ ] [Concept 1]
- [ ] [Concept 2]
- [ ] [Concept 3]

**If any rating is below 3, review that section before continuing.**

---

### 🏋️ Challenge 1: Guided Practice (Completion Problem)

**Scenario:** [Describe realistic problem]

**Your task:**
[Partially completed code with blanks]

<details>
<summary>Show solution</summary>

[Complete solution with explanation]

</details>

---

### 🚀 Challenge 2: Similar Problem with Hints

**Scenario:** [New but similar problem]

**Hints:**
- [Hint 1]
- [Hint 2]

[No code provided - they implement from scratch]

<details>
<summary>Show solution</summary>

[Solution]

</details>

---

### 💪 Challenge 3: Independent Application (Transfer)

**Scenario:** [Novel problem requiring transfer]

**Your task:** [Describe task - no hints, no code scaffolding]

[NO SOLUTION PROVIDED - This is independent practice]

**Discussion:** Share your solution in the forum!

---

### 📊 Theory Questions

**Conceptual:**
1. [Deep understanding question]
2. [Comparison question]

**Computational:**
3. [Hand calculation problem]

**Critical Thinking:**
4. [When would this fail?]
5. [Compare to another algorithm]

<details>
<summary>Show answer guide</summary>

[Answers with detailed explanations]

</details>

---

### 🔄 Spaced Review Schedule

**To maximize retention:**
- ✅ Today: Complete all challenges above
- 📅 Tomorrow: Quick quiz (5 min)
- 📅 Next week: Mixed practice
- 📅 Next month: Cumulative review

---

### 🚀 Next Steps

**Ready to move on?** You should be able to:
- [Success criterion 1]
- [Success criterion 2]
- [Success criterion 3]

**Need more practice?**
- [Resource 1]
- [Resource 2]

**Continue to:** [Next tutorial]`}</code></pre>

      <hr />

      <h2>Worked Example Progression Pattern</h2>

      <p>Use this 4-step progression for teaching complex skills:</p>

      <h3>Step 1: Fully Worked Example (Current Standard ✓)</h3>
      <pre><code>{`# Show complete solution with detailed comments
def complete_example():
    # Full implementation with explanation
    pass`}</code></pre>

      <h3>Step 2: Completion Problem (ADD THIS)</h3>
      <pre><code>{`# Your Turn: Fill in the blanks
def completion_exercise():
    result = _____  # Hint provided
    return result`}</code></pre>

      <h3>Step 3: Guided Problem (ADD THIS)</h3>
      <pre><code>{`Similar problem with hints but no code scaffolding
Hints:
- [Hint 1]
- [Hint 2]`}</code></pre>

      <h3>Step 4: Independent Problem (ADD THIS)</h3>
      <pre><code>{`Novel application - no hints, no scaffolding
Learner implements from scratch`}</code></pre>

      <hr />

      <h2>Quality Standards for New Elements</h2>

      <h3>Good Retrieval Questions:</h3>
      <ul>
        <li>✅ Require recall, not recognition</li>
        <li>✅ Target key concepts, not trivial details</li>
        <li>✅ Include both factual and conceptual questions</li>
        <li>❌ Avoid trick questions or obscure edge cases</li>
      </ul>

      <h3>Good Completion Problems:</h3>
      <ul>
        <li>✅ Remove 20-40% of the code (not too much or too little)</li>
        <li>✅ Provide meaningful hints</li>
        <li>✅ Focus on concepts, not syntax</li>
        <li>❌ Don't just remove variable names</li>
      </ul>

      <h3>Good Misconceptions:</h3>
      <ul>
        <li>✅ Address actual misconceptions learners have</li>
        <li>✅ Explain why the misconception is tempting</li>
        <li>✅ Provide concrete examples showing the difference</li>
        <li>❌ Don't create strawman arguments no one actually believes</li>
      </ul>

      <h3>Good Reflection Prompts:</h3>
      <ul>
        <li>✅ Open-ended, requiring thinking</li>
        <li>✅ Connect to prior knowledge</li>
        <li>✅ Encourage metacognition</li>
        <li>❌ Avoid yes/no questions</li>
      </ul>

      <hr />

      <h2>Spaced Repetition System</h2>

      <h3>Immediate Review (End of Tutorial)</h3>
      <pre><code>Complete all "Quick Check" questions and practice challenges</code></pre>

      <h3>Day 1 Review (Next Day)</h3>
      <pre><code>{`### 5-Minute Quick Quiz

Without looking at notes:
1. [Key concept question]
2. [Application question]
3. [Comparison question]

[Link to answers]`}</code></pre>

      <h3>Week 1 Review (One Week Later)</h3>
      <pre><code>{`### Mixed Practice (15 minutes)

This mixes concepts from [Tutorial A] and [Tutorial B]:

1. [Question mixing both topics]
2. [Comparison between algorithms]
3. [Which algorithm for this scenario?]`}</code></pre>

      <h3>Month 1 Review (One Month Later)</h3>
      <pre><code>{`### Cumulative Assessment

[Comprehensive quiz covering all tutorials completed]
[Transfer problems with novel datasets]`}</code></pre>

      <hr />

      <h2>Priority Matrix: What to Add First</h2>

      <h3>CRITICAL (Do First - Highest Impact per Hour)</h3>
      <ol>
        <li><strong>Retrieval practice</strong> - 3 "Quick Check" sections per tutorial (2-3 hours per tutorial)</li>
        <li><strong>Misconceptions</strong> - Address 3-5 per tutorial (1-2 hours per tutorial)</li>
        <li><strong>End-of-tutorial challenges</strong> - Guided to independent progression (3-4 hours per tutorial)</li>
      </ol>

      <h3>HIGH (Do Second)</h3>
      <ol start={4}>
        <li><strong>Learning objectives</strong> - Before each major section (1 hour per tutorial)</li>
        <li><strong>Reflection checkpoints</strong> - After complex sections (1 hour per tutorial)</li>
        <li><strong>Completion problems</strong> - After worked examples (2-3 hours per tutorial)</li>
      </ol>

      <h3>MEDIUM (Do Third)</h3>
      <ol start={7}>
        <li><strong>Prediction prompts</strong> - Before code outputs (1 hour per tutorial)</li>
        <li><strong>Spaced retrieval schedule</strong> - Daily/weekly/monthly quizzes (8-10 hours total)</li>
        <li><strong>Multiple contexts</strong> - 2-3 datasets per algorithm (5-6 hours per tutorial)</li>
      </ol>

      <h3>LOW (Nice to Have)</h3>
      <ol start={10}>
        <li><strong>Pre-assessments</strong> - Personalized paths (5-6 hours)</li>
        <li><strong>Growth mindset messaging</strong> - Normalize struggle (1 hour per tutorial)</li>
        <li><strong>CRA sequences</strong> - For especially complex topics (2-3 hours per topic)</li>
      </ol>

      <hr />

      <h2>Measuring Success</h2>

      <h3>Before Changes (Baseline)</h3>
      <ul>
        <li>Quiz performance after 1 week</li>
        <li>Quiz performance after 1 month</li>
        <li>Completion rates</li>
        <li>Time to complete</li>
        <li>Self-reported confidence</li>
      </ul>

      <h3>After Changes (Expected Improvements)</h3>
      <ul>
        <li>+20-30% on delayed retention (1 month)</li>
        <li>+15-25% on transfer tasks</li>
        <li>+10-15% completion rates</li>
        <li>+25-35% self-reported confidence</li>
      </ul>

      <h3>A/B Testing Approach</h3>
      <ul>
        <li><strong>Group A:</strong> Current passive format</li>
        <li><strong>Group B:</strong> Enhanced with retrieval practice</li>
        <li><strong>Measure:</strong> 1-week retention quiz performance</li>
      </ul>

      <hr />

      <h2>Getting Started: Your First Enhancement</h2>

      <p><strong>Recommended Pilot:</strong> Linear Regression tutorial</p>

      <p><strong>Timeline:</strong> 1 week</p>

      <p><strong>Steps:</strong></p>
      <ol>
        <li><strong>Day 1-2:</strong> Add retrieval practice (3 Quick Check sections)</li>
        <li><strong>Day 3-4:</strong> Address misconceptions (5 Common Pitfalls boxes)</li>
        <li><strong>Day 5-6:</strong> Add end-of-tutorial challenges (3 progressive challenges)</li>
        <li><strong>Day 7:</strong> Test with 5-10 learners, gather feedback</li>
      </ol>

      <p><strong>Measure:</strong></p>
      <ul>
        <li>Give learners a quiz 1 week after completion</li>
        <li>Compare to baseline (if available) or expected benchmarks</li>
        <li>Interview 2-3 learners about their experience</li>
      </ul>

      <p><strong>Iterate:</strong></p>
      <ul>
        <li>Refine based on feedback</li>
        <li>Scale to other tutorials</li>
        <li>Continuously improve</li>
      </ul>

      <hr />

      <h2>Additional Resources</h2>

      <h3>Quick Reference Books</h3>
      <ul>
        <li><strong>Make It Stick</strong> by Brown, Roediger, & McDaniel - Most accessible introduction</li>
        <li><strong>Small Teaching</strong> by Lang - Practical, evidence-based strategies</li>
      </ul>

      <h3>Research-Backed Techniques</h3>
      <ul>
        <li><strong>Retrieval Practice:</strong> Testing yourself strengthens memory more than re-reading</li>
        <li><strong>Spaced Repetition:</strong> Review at increasing intervals for long-term retention</li>
        <li><strong>Interleaving:</strong> Mix different topics rather than blocking by topic</li>
        <li><strong>Elaboration:</strong> Explain concepts in your own words</li>
        <li><strong>Concrete Examples:</strong> Start with specific cases before general principles</li>
      </ul>

      <h3>Learning Science Principles</h3>
      <ol>
        <li><strong>Cognitive Load Theory:</strong> Don't overwhelm working memory</li>
        <li><strong>Dual Coding:</strong> Combine words and visuals</li>
        <li><strong>Worked Examples:</strong> Show before asking</li>
        <li><strong>Desirable Difficulties:</strong> Some struggle aids learning</li>
        <li><strong>Metacognition:</strong> Think about your thinking</li>
      </ol>

      <hr />

      <h2>Implementation Checklist for Each Tutorial</h2>

      <p>Before considering a tutorial "enhanced":</p>

      <h3>Structure:</h3>
      <ul>
        <li>[ ] Learning objectives before each major section</li>
        <li>[ ] Prediction prompts before code outputs</li>
        <li>[ ] Retrieval questions after each major section</li>
        <li>[ ] Reflection checkpoints after complex topics</li>
        <li>[ ] Comprehensive end-of-tutorial challenges</li>
      </ul>

      <h3>Content:</h3>
      <ul>
        <li>[ ] 3-5 common misconceptions addressed</li>
        <li>[ ] Worked example progression (complete → completion → guided → independent)</li>
        <li>[ ] Multiple modalities (text, code, visuals, math)</li>
        <li>[ ] Connection to real-world applications</li>
      </ul>

      <h3>Assessment:</h3>
      <ul>
        <li>[ ] Self-assessment opportunities throughout</li>
        <li>[ ] Progressive practice (easy → hard)</li>
        <li>[ ] Both factual and conceptual questions</li>
        <li>[ ] Transfer problems (novel applications)</li>
      </ul>

      <h3>Learner Support:</h3>
      <ul>
        <li>[ ] Prerequisites clearly stated</li>
        <li>[ ] Hints available (but hidden)</li>
        <li>[ ] Struggle normalized with support strategies</li>
        <li>[ ] Multiple paths (quick review vs. deep dive)</li>
      </ul>

      <hr />

      <h2>Theoretical Foundation</h2>

      <p>All recommendations based on research in:</p>
      <ul>
        <li>Cognitive psychology (how the brain learns)</li>
        <li>Educational psychology (how to teach effectively)</li>
        <li>Learning science (empirical studies of learning)</li>
      </ul>

      <p>See <strong>LEARNING_SCIENCE_REVIEW.md</strong> for complete research citations and detailed analysis.</p>

      <p><strong>Key Principle:</strong> Active engagement beats passive consumption. Make learners DO, not just READ.</p>

      <hr />

      <p><strong>Last Updated:</strong> 2025-10-24</p>
      <p><strong>See Also:</strong> LEARNING_SCIENCE_REVIEW.md, PROJECT_GUIDELINES.md</p>
      </article>
    </SidebarLayoutContent>
  );
}
