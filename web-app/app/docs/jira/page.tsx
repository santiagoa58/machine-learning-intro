import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: "Task Tracker",
  description: "Project roadmap, sprint plans, and task tracking for the ML tutorial platform development.",
  openGraph: {
    title: "Task Tracker | ML Introduction",
    description: "Project roadmap, sprint plans, and task tracking for the ML tutorial platform development.",
    type: "article",
  },
};

export default function JiraPage() {
  return (
    <article>
      <h1>Machine Learning Tutorial Platform - Task Tracker</h1>

      <p>
        <strong>Project:</strong> ML Learning Platform<br />
        <strong>Last Updated:</strong> 2025-10-25<br />
        <strong>Current Sprint:</strong> Sprint 0 (Planning & Setup)
      </p>

      <hr />

      <h2>📊 Project Overview</h2>

      <h3>Vision</h3>
      <p>
        Transform ML learning from passive Jupyter notebooks to an interactive web platform that implements evidence-based learning science principles for maximum retention and mastery.
      </p>

      <h3>Goals</h3>
      <ul>
        <li><strong>Primary:</strong> Launch interactive web app with core ML tutorials</li>
        <li><strong>Secondary:</strong> Implement spaced repetition and progress tracking</li>
        <li><strong>Tertiary:</strong> Build community features and advanced analytics</li>
      </ul>

      <h3>Tech Stack (Latest Stable Versions)</h3>
      <ul>
        <li><strong>Frontend:</strong> Next.js 16.0.0, React 19.2.0, TypeScript 5</li>
        <li><strong>Styling:</strong> Tailwind CSS v4 (using Compass template from Tailwind Plus)</li>
        <li><strong>Bundler:</strong> Turbopack (Next.js built-in, replaces webpack)</li>
        <li><strong>Content:</strong> MDX for interactive tutorials</li>
        <li><strong>Backend:</strong> Next.js API routes (App Router)</li>
        <li><strong>Database:</strong> (TBD - PostgreSQL/Supabase for user progress)</li>
        <li><strong>Deployment:</strong> Vercel</li>
        <li><strong>Code Execution:</strong> Pyodide (Python in browser)</li>
      </ul>

      <hr />

      <h2>📈 Sprint Status</h2>

      <h3>Sprint 0: Planning & Foundation (Current)</h3>
      <ul>
        <li><strong>Duration:</strong> Week 1-2</li>
        <li><strong>Goal:</strong> Complete project setup, architecture design, content audit</li>
        <li><strong>Status:</strong> 🟡 In Progress</li>
      </ul>

      <h3>Sprint 1: Core Web App Infrastructure</h3>
      <ul>
        <li><strong>Duration:</strong> Week 3-4</li>
        <li><strong>Goal:</strong> Next.js app with Compass template, basic navigation, first tutorial</li>
        <li><strong>Status:</strong> ⚪ Not Started</li>
      </ul>

      <h3>Sprint 2: Interactive Learning Features</h3>
      <ul>
        <li><strong>Duration:</strong> Week 5-6</li>
        <li><strong>Goal:</strong> Implement retrieval practice, completion problems, code execution</li>
        <li><strong>Status:</strong> ⚪ Not Started</li>
      </ul>

      <h3>Sprint 3: Content Migration & Enhancement</h3>
      <ul>
        <li><strong>Duration:</strong> Week 7-8</li>
        <li><strong>Goal:</strong> Migrate all tutorials, add learning science elements</li>
        <li><strong>Status:</strong> ⚪ Not Started</li>
      </ul>

      <h3>Sprint 4: Progress Tracking & Spaced Repetition</h3>
      <ul>
        <li><strong>Duration:</strong> Week 9-10</li>
        <li><strong>Goal:</strong> User accounts, progress tracking, spaced review system</li>
        <li><strong>Status:</strong> ⚪ Not Started</li>
      </ul>

      <h3>Sprint 5: Polish & Launch</h3>
      <ul>
        <li><strong>Duration:</strong> Week 11-12</li>
        <li><strong>Goal:</strong> Testing, optimization, public launch</li>
        <li><strong>Status:</strong> ⚪ Not Started</li>
      </ul>

      <hr />

      <h2>🎯 Epics</h2>

      <h3>EPIC-1: Project Documentation & Guidelines ✅ COMPLETE</h3>
      <p>
        <strong>Status:</strong> Done<br />
        <strong>Priority:</strong> Critical<br />
        <strong>Completed:</strong> 2025-10-25
      </p>
      <p>Stories:</p>
      <ul>
        <li>[STORY-1] Create README.md - ✅ Done</li>
        <li>[STORY-2] Create PROJECT_GUIDELINES.md - ✅ Done</li>
        <li>[STORY-3] Learning Science Review - ✅ Done</li>
        <li>[STORY-4] Improvement Guide - ✅ Done</li>
      </ul>

      <hr />

      <h3>EPIC-2: Content Quality & Accuracy</h3>
      <p>
        <strong>Status:</strong> In Progress (80% complete)<br />
        <strong>Priority:</strong> Critical<br />
        <strong>Target:</strong> End of Sprint 0
      </p>
      <p><strong>Description:</strong> Ensure all existing notebook content is technically accurate, follows guidelines, and is ready for migration to web platform.</p>
      <p><strong>Progress:</strong> 7/11 critical fixes complete, HIGH priority fixes done</p>

      <h4>Stories</h4>

      <h4>[STORY-5] Fix Critical Technical Inaccuracies ✅ COMPLETE</h4>
      <ul>
        <li>Status: Done</li>
        <li>Priority: Critical</li>
        <li>Tasks:
          <ul>
            <li>[TASK-1] ✅ Fix KNN computational complexity (O(N²) → O(N·d))</li>
            <li>[TASK-2] ✅ Fix KNN dimensionality characterization</li>
            <li>[TASK-3] ✅ Fix Linear Regression R² explanation</li>
            <li>[TASK-4] ✅ Add random_state to all Linear Regression splits</li>
            <li>[TASK-5] ✅ Fix Logistic Regression confusion matrix</li>
            <li>[TASK-6] ✅ Fix Logistic Regression sigmoid domain/range</li>
            <li>[TASK-7] ✅ Fix SVM predict_similarity function</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-6] Fix Guideline Adherence Issues ✅ COMPLETE</h4>
      <ul>
        <li>Status: Done</li>
        <li>Priority: High</li>
        <li>Tasks:
          <ul>
            <li>[TASK-8] ✅ Fix duplicate section numbering in Linear Regression</li>
            <li>[TASK-9] ✅ Add random_state to all Logistic Regression models (6 locations)</li>
            <li>[TASK-10] ✅ Fix typo "Accurary" → "Accuracy"</li>
            <li>[TASK-11] ✅ Fix duplicate section 2.4.1 in KNN</li>
            <li>[TASK-12] ✅ Add random_state to SVM.py (SVC and PCA)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-7] Complete Foundation Tutorials</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 21</li>
        <li>Description: Write complete tutorials for numpy, pandas, sklearn</li>
        <li>Tasks:
          <ul>
            <li>[TASK-13] ⚪ Write numpy.ipynb following PROJECT_GUIDELINES.md (7 points)</li>
            <li>[TASK-14] ⚪ Write pandas.ipynb following PROJECT_GUIDELINES.md (7 points)</li>
            <li>[TASK-15] ⚪ Write sklearn.ipynb following PROJECT_GUIDELINES.md (7 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-8] Add Missing Content Sections</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 13</li>
        <li>Description: Add sections referenced in existing tutorials but not yet written</li>
        <li>Tasks:
          <ul>
            <li>[TASK-16] ⚪ Add KNN distance metrics theory (sections 3.1-3.4) (5 points)</li>
            <li>[TASK-17] ⚪ Add KNN curse of dimensionality detailed section (3 points)</li>
            <li>[TASK-18] ⚪ Add Linear Regression assumptions section (3 points)</li>
            <li>[TASK-19] ⚪ Add Linear Regression polynomial degree explanation (2 points)</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h3>EPIC-3: Web Application Infrastructure</h3>
      <p>
        <strong>Status:</strong> To Do<br />
        <strong>Priority:</strong> Critical<br />
        <strong>Target:</strong> End of Sprint 1
      </p>
      <p><strong>Description:</strong> Build Next.js web application with Tailwind Compass template, basic navigation, and architecture for interactive learning.</p>

      <h4>Stories</h4>

      <h4>[STORY-9] Project Setup & Configuration</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Critical</li>
        <li>Story Points: 5</li>
        <li>Sprint: Sprint 1</li>
        <li>Tasks:
          <ul>
            <li>[TASK-20] ⚪ Initialize Next.js 16 project with TypeScript (1 point)</li>
            <li>[TASK-21] ⚪ Set up Tailwind CSS with Compass template integration (2 points)</li>
            <li>[TASK-22] ⚪ Configure project structure (app/, components/, lib/, etc.) (1 point)</li>
            <li>[TASK-23] ⚪ Set up ESLint, Prettier, Git hooks (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-10] Implement Compass Template Design</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Critical</li>
        <li>Story Points: 8</li>
        <li>Sprint: Sprint 1</li>
        <li>Description: Adapt Tailwind Plus Compass template for ML learning platform</li>
        <li>Tasks:
          <ul>
            <li>[TASK-24] ⚪ Extract and integrate Compass theme components (2 points)</li>
            <li>[TASK-25] ⚪ Create main navigation layout (2 points)</li>
            <li>[TASK-26] ⚪ Implement responsive sidebar for tutorial navigation (2 points)</li>
            <li>[TASK-27] ⚪ Design homepage with learning path visualization (2 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-11] Tutorial Content System</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Critical</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 1</li>
        <li>Description: Build system for rendering MDX tutorials with interactive elements</li>
        <li>Tasks:
          <ul>
            <li>[TASK-28] ⚪ Set up MDX integration with Next.js (3 points)</li>
            <li>[TASK-29] ⚪ Create tutorial page template with TOC (3 points)</li>
            <li>[TASK-30] ⚪ Implement syntax highlighting for code blocks (2 points)</li>
            <li>[TASK-31] ⚪ Add LaTeX math rendering (KaTeX or similar) (2 points)</li>
            <li>[TASK-32] ⚪ Create component library for tutorial elements (3 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-12] Code Execution Environment</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 1</li>
        <li>Description: Enable Python code execution in browser using Pyodide</li>
        <li>Tasks:
          <ul>
            <li>[TASK-33] ⚪ Integrate Pyodide for browser-based Python (5 points)</li>
            <li>[TASK-34] ⚪ Create code editor component (CodeMirror/Monaco) (3 points)</li>
            <li>[TASK-35] ⚪ Implement run/execute functionality (2 points)</li>
            <li>[TASK-36] ⚪ Add output console display (2 points)</li>
            <li>[TASK-37] ⚪ Handle matplotlib/plotting in browser (1 point)</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h3>EPIC-4: Interactive Learning Features</h3>
      <p>
        <strong>Status:</strong> To Do<br />
        <strong>Priority:</strong> Critical<br />
        <strong>Target:</strong> End of Sprint 2
      </p>
      <p><strong>Description:</strong> Implement learning science principles as interactive web features (retrieval practice, completion problems, feedback).</p>

      <h4>Stories</h4>

      <h4>[STORY-13] Retrieval Practice Components</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Critical</li>
        <li>Story Points: 8</li>
        <li>Sprint: Sprint 2</li>
        <li>Description: Build interactive "Quick Check" components with reveal answers</li>
        <li>Tasks:
          <ul>
            <li>[TASK-38] ⚪ Create QuickCheck component with collapsible answers (3 points)</li>
            <li>[TASK-39] ⚪ Build multi-choice question component (2 points)</li>
            <li>[TASK-40] ⚪ Create fill-in-the-blank component (2 points)</li>
            <li>[TASK-41] ⚪ Add immediate feedback mechanism (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-14] Completion Problem System</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Critical</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 2</li>
        <li>Description: Interactive code completion exercises with hints and solutions</li>
        <li>Tasks:
          <ul>
            <li>[TASK-42] ⚪ Create CodeCompletion component (blank detection) (5 points)</li>
            <li>[TASK-43] ⚪ Implement hint system (progressive reveals) (3 points)</li>
            <li>[TASK-44] ⚪ Add solution comparison/checking (3 points)</li>
            <li>[TASK-45] ⚪ Create visual feedback (correct/incorrect indicators) (2 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-15] Prediction Prompts</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 5</li>
        <li>Sprint: Sprint 2</li>
        <li>Description: Components that ask learners to predict before revealing</li>
        <li>Tasks:
          <ul>
            <li>[TASK-46] ⚪ Create PredictionPrompt component (2 points)</li>
            <li>[TASK-47] ⚪ Build comparison visualization (prediction vs actual) (2 points)</li>
            <li>[TASK-48] ⚪ Add reflection questions after reveal (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-16] Misconception Alerts</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 3</li>
        <li>Sprint: Sprint 2</li>
        <li>Description: Callout components for common misconceptions</li>
        <li>Tasks:
          <ul>
            <li>[TASK-49] ⚪ Create Misconception component with styling (2 points)</li>
            <li>[TASK-50] ⚪ Add self-check questions (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-17] Reflection Checkpoints</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 5</li>
        <li>Sprint: Sprint 2</li>
        <li>Description: Metacognitive reflection components</li>
        <li>Tasks:
          <ul>
            <li>[TASK-51] ⚪ Create Reflection component (2 points)</li>
            <li>[TASK-52] ⚪ Build confidence rating widget (2 points)</li>
            <li>[TASK-53] ⚪ Add self-assessment checklist (1 point)</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h3>EPIC-5: Content Migration & Enhancement</h3>
      <p>
        <strong>Status:</strong> To Do<br />
        <strong>Priority:</strong> High<br />
        <strong>Target:</strong> End of Sprint 3
      </p>
      <p><strong>Description:</strong> Convert Jupyter notebooks to MDX with interactive learning science elements.</p>

      <h4>Stories</h4>

      <h4>[STORY-18] Convert Linear Regression to MDX</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Critical</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 3</li>
        <li>Description: Migrate Linear Regression notebook with all learning enhancements</li>
        <li>Tasks:
          <ul>
            <li>[TASK-54] ⚪ Convert notebook content to MDX format (3 points)</li>
            <li>[TASK-55] ⚪ Add 5 Quick Check sections (3 points)</li>
            <li>[TASK-56] ⚪ Add 3 completion problems (3 points)</li>
            <li>[TASK-57] ⚪ Add 5 common misconceptions (2 points)</li>
            <li>[TASK-58] ⚪ Add prediction prompts (1 point)</li>
            <li>[TASK-59] ⚪ Add end-of-tutorial assessment (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-19] Convert KNN to MDX</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 3</li>
        <li>Tasks:
          <ul>
            <li>[TASK-60] ⚪ Convert notebook content to MDX format (3 points)</li>
            <li>[TASK-61] ⚪ Add 5 Quick Check sections (3 points)</li>
            <li>[TASK-62] ⚪ Add 3 completion problems (3 points)</li>
            <li>[TASK-63] ⚪ Add 5 common misconceptions (2 points)</li>
            <li>[TASK-64] ⚪ Add prediction prompts (1 point)</li>
            <li>[TASK-65] ⚪ Add end-of-tutorial assessment (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-20] Convert Logistic Regression to MDX</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 3</li>
        <li>Tasks:
          <ul>
            <li>[TASK-66] ⚪ Convert notebook content to MDX format (3 points)</li>
            <li>[TASK-67] ⚪ Add 5 Quick Check sections (3 points)</li>
            <li>[TASK-68] ⚪ Add 3 completion problems (3 points)</li>
            <li>[TASK-69] ⚪ Add 5 common misconceptions (2 points)</li>
            <li>[TASK-70] ⚪ Add prediction prompts (1 point)</li>
            <li>[TASK-71] ⚪ Add end-of-tutorial assessment (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-21] Convert SVM to MDX</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 3</li>
        <li>Description: Create SVM tutorial (currently only has .py file) with full learning elements</li>
        <li>Tasks:
          <ul>
            <li>[TASK-72] ⚪ Write SVM tutorial from scratch in MDX (5 points)</li>
            <li>[TASK-73] ⚪ Add 5 Quick Check sections (3 points)</li>
            <li>[TASK-74] ⚪ Add 3 completion problems (3 points)</li>
            <li>[TASK-75] ⚪ Add 5 common misconceptions (2 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-22] Create Foundation Tutorials in MDX</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 21</li>
        <li>Sprint: Sprint 3</li>
        <li>Tasks:
          <ul>
            <li>[TASK-76] ⚪ Create numpy tutorial in MDX (7 points)</li>
            <li>[TASK-77] ⚪ Create pandas tutorial in MDX (7 points)</li>
            <li>[TASK-78] ⚪ Create sklearn tutorial in MDX (7 points)</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h3>EPIC-6: Progress Tracking & User System</h3>
      <p>
        <strong>Status:</strong> To Do<br />
        <strong>Priority:</strong> High<br />
        <strong>Target:</strong> End of Sprint 4
      </p>
      <p><strong>Description:</strong> User authentication, progress tracking, and personalized learning paths.</p>

      <h4>Stories</h4>

      <h4>[STORY-23] User Authentication</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 8</li>
        <li>Sprint: Sprint 4</li>
        <li>Tasks:
          <ul>
            <li>[TASK-79] ⚪ Set up authentication (NextAuth.js or similar) (3 points)</li>
            <li>[TASK-80] ⚪ Implement sign-up/login UI (2 points)</li>
            <li>[TASK-81] ⚪ Create user profile page (2 points)</li>
            <li>[TASK-82] ⚪ Add session management (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-24] Progress Tracking System</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 4</li>
        <li>Description: Track completion, quiz scores, time spent per tutorial</li>
        <li>Tasks:
          <ul>
            <li>[TASK-83] ⚪ Design database schema for user progress (2 points)</li>
            <li>[TASK-84] ⚪ Set up database (Supabase/PostgreSQL) (3 points)</li>
            <li>[TASK-85] ⚪ Implement progress tracking API (3 points)</li>
            <li>[TASK-86] ⚪ Create progress visualization dashboard (3 points)</li>
            <li>[TASK-87] ⚪ Add "resume where you left off" functionality (2 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-25] Achievement System</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 8</li>
        <li>Sprint: Sprint 4</li>
        <li>Description: Badges, streaks, completion milestones</li>
        <li>Tasks:
          <ul>
            <li>[TASK-88] ⚪ Define achievement types and triggers (2 points)</li>
            <li>[TASK-89] ⚪ Create achievement UI components (3 points)</li>
            <li>[TASK-90] ⚪ Implement achievement logic (2 points)</li>
            <li>[TASK-91] ⚪ Add achievement notifications (1 point)</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h3>EPIC-7: Spaced Repetition System</h3>
      <p>
        <strong>Status:</strong> To Do<br />
        <strong>Priority:</strong> Medium<br />
        <strong>Target:</strong> End of Sprint 4
      </p>
      <p><strong>Description:</strong> Implement spaced repetition for long-term retention using research-backed intervals.</p>

      <h4>Stories</h4>

      <h4>[STORY-26] Quiz Generation System</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 4</li>
        <li>Tasks:
          <ul>
            <li>[TASK-92] ⚪ Create quiz question bank structure (3 points)</li>
            <li>[TASK-93] ⚪ Build quiz component (multiple choice, fill-in, code) (5 points)</li>
            <li>[TASK-94] ⚪ Implement quiz scoring/feedback (2 points)</li>
            <li>[TASK-95] ⚪ Create quiz result analytics (3 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-27] Spaced Review Schedule</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 4</li>
        <li>Description: Algorithm to schedule reviews at 1 day, 1 week, 1 month intervals</li>
        <li>Tasks:
          <ul>
            <li>[TASK-96] ⚪ Implement SM-2 or similar spaced repetition algorithm (5 points)</li>
            <li>[TASK-97] ⚪ Create review scheduling system (3 points)</li>
            <li>[TASK-98] ⚪ Build "due for review" notification system (3 points)</li>
            <li>[TASK-99] ⚪ Create daily review page (2 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-28] Cumulative Review Notebooks</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Low</li>
        <li>Story Points: 8</li>
        <li>Sprint: Sprint 4</li>
        <li>Description: Mixed practice combining multiple tutorials</li>
        <li>Tasks:
          <ul>
            <li>[TASK-100] ⚪ Design cumulative review format (2 points)</li>
            <li>[TASK-101] ⚪ Create first cumulative review (Linear Reg + KNN) (3 points)</li>
            <li>[TASK-102] ⚪ Create second cumulative review (all supervised learning) (3 points)</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h3>EPIC-8: Deployment & Infrastructure</h3>
      <p>
        <strong>Status:</strong> To Do<br />
        <strong>Priority:</strong> High<br />
        <strong>Target:</strong> End of Sprint 5
      </p>
      <p><strong>Description:</strong> Deploy to production, set up analytics, monitoring, and CI/CD.</p>

      <h4>Stories</h4>

      <h4>[STORY-29] Vercel Deployment</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Critical</li>
        <li>Story Points: 5</li>
        <li>Sprint: Sprint 5</li>
        <li>Tasks:
          <ul>
            <li>[TASK-103] ⚪ Configure Vercel project (1 point)</li>
            <li>[TASK-104] ⚪ Set up environment variables (1 point)</li>
            <li>[TASK-105] ⚪ Configure custom domain (1 point)</li>
            <li>[TASK-106] ⚪ Set up preview deployments (1 point)</li>
            <li>[TASK-107] ⚪ Configure build optimization (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-30] Analytics & Monitoring</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 8</li>
        <li>Sprint: Sprint 5</li>
        <li>Tasks:
          <ul>
            <li>[TASK-108] ⚪ Set up analytics (Vercel Analytics or Google Analytics) (2 points)</li>
            <li>[TASK-109] ⚪ Implement error tracking (Sentry) (2 points)</li>
            <li>[TASK-110] ⚪ Add performance monitoring (2 points)</li>
            <li>[TASK-111] ⚪ Create admin dashboard for metrics (2 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-31] CI/CD Pipeline</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 5</li>
        <li>Sprint: Sprint 5</li>
        <li>Tasks:
          <ul>
            <li>[TASK-112] ⚪ Set up GitHub Actions (2 points)</li>
            <li>[TASK-113] ⚪ Add automated testing (1 point)</li>
            <li>[TASK-114] ⚪ Configure linting/type checking in CI (1 point)</li>
            <li>[TASK-115] ⚪ Add automated deployment on merge to main (1 point)</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h3>EPIC-9: Polish & Launch Preparation</h3>
      <p>
        <strong>Status:</strong> To Do<br />
        <strong>Priority:</strong> High<br />
        <strong>Target:</strong> End of Sprint 5
      </p>
      <p><strong>Description:</strong> Final polish, testing, documentation, and launch.</p>

      <h4>Stories</h4>

      <h4>[STORY-32] Accessibility & SEO</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 8</li>
        <li>Sprint: Sprint 5</li>
        <li>Tasks:
          <ul>
            <li>[TASK-116] ⚪ ARIA labels and keyboard navigation (3 points)</li>
            <li>[TASK-117] ⚪ Color contrast and screen reader testing (2 points)</li>
            <li>[TASK-118] ⚪ Meta tags and OpenGraph (2 points)</li>
            <li>[TASK-119] ⚪ Sitemap and robots.txt (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-33] Performance Optimization</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: High</li>
        <li>Story Points: 8</li>
        <li>Sprint: Sprint 5</li>
        <li>Tasks:
          <ul>
            <li>[TASK-120] ⚪ Image optimization (next/image) (2 points)</li>
            <li>[TASK-121] ⚪ Code splitting and lazy loading (2 points)</li>
            <li>[TASK-122] ⚪ Bundle size analysis and reduction (2 points)</li>
            <li>[TASK-123] ⚪ Lighthouse audit and fixes (2 points)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-34] User Testing</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Critical</li>
        <li>Story Points: 13</li>
        <li>Sprint: Sprint 5</li>
        <li>Tasks:
          <ul>
            <li>[TASK-124] ⚪ Recruit 10-15 beta testers (2 points)</li>
            <li>[TASK-125] ⚪ Conduct user testing sessions (5 points)</li>
            <li>[TASK-126] ⚪ Analyze feedback and create bug list (2 points)</li>
            <li>[TASK-127] ⚪ Fix critical bugs from testing (3 points)</li>
            <li>[TASK-128] ⚪ Iterate on UX issues (1 point)</li>
          </ul>
        </li>
      </ul>

      <h4>[STORY-35] Launch Materials</h4>
      <ul>
        <li>Status: To Do</li>
        <li>Priority: Medium</li>
        <li>Story Points: 5</li>
        <li>Sprint: Sprint 5</li>
        <li>Tasks:
          <ul>
            <li>[TASK-129] ⚪ Create launch video/demo (2 points)</li>
            <li>[TASK-130] ⚪ Write blog post announcement (1 point)</li>
            <li>[TASK-131] ⚪ Prepare social media content (1 point)</li>
            <li>[TASK-132] ⚪ Submit to relevant directories (HN, Reddit, etc.) (1 point)</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h2>🐛 Known Issues</h2>

      <h3>Critical Bugs</h3>
      <p>None currently</p>

      <h3>High Priority Bugs</h3>
      <p>None currently</p>

      <h3>Medium Priority Bugs</h3>
      <p>None currently</p>

      <h3>Low Priority Bugs</h3>
      <ul>
        <li>[BUG-1] ⚪ Image attachments in notebooks won't display (affects all notebooks)
          <ul>
            <li>Priority: Low</li>
            <li>Description: Notebooks reference attachment:image.png which won't render</li>
            <li>Fix: Need to extract images and host properly or use external URLs</li>
            <li>Affected: Linear Regression, Linear Regression Deeper Dive, KNN</li>
          </ul>
        </li>
      </ul>

      <hr />

      <h2>📝 Backlog (Future Enhancements)</h2>

      <h3>Phase 2 Features (Post-Launch)</h3>

      <h4>[EPIC-10] Community Features</h4>
      <ul>
        <li>Discussion forums</li>
        <li>Peer learning/study groups</li>
        <li>Leaderboards</li>
        <li>User-submitted content</li>
      </ul>

      <h4>[EPIC-11] Advanced Algorithms</h4>
      <ul>
        <li>Decision Trees</li>
        <li>Random Forests</li>
        <li>Neural Networks (intro)</li>
        <li>Clustering algorithms</li>
        <li>Dimensionality reduction</li>
      </ul>

      <h4>[EPIC-12] Interactive Visualizations</h4>
      <ul>
        <li>Algorithm animations (gradient descent, KNN search, etc.)</li>
        <li>Interactive plots (adjust parameters and see results)</li>
        <li>Decision boundary visualizations</li>
        <li>3D visualizations for higher-dimensional concepts</li>
      </ul>

      <h4>[EPIC-13] Mobile App</h4>
      <ul>
        <li>React Native mobile app</li>
        <li>Offline mode</li>
        <li>Push notifications for spaced reviews</li>
      </ul>

      <h4>[EPIC-14] Advanced Analytics</h4>
      <ul>
        <li>Learning path recommendations based on performance</li>
        <li>Difficulty adaptation</li>
        <li>Personalized review schedules</li>
        <li>Predictive analytics for completion</li>
      </ul>

      <h4>[EPIC-15] Enterprise Features</h4>
      <ul>
        <li>Classroom management</li>
        <li>Assignment creation</li>
        <li>Grading tools</li>
        <li>Team accounts</li>
      </ul>

      <hr />

      <h2>🎯 Current Sprint Details</h2>

      <h3>Sprint 0: Planning & Foundation (Current)</h3>

      <p><strong>Sprint Goal:</strong> Complete all planning, documentation, and prepare for web app development</p>

      <p><strong>Sprint Tasks:</strong></p>
      <ul>
        <li>[TASK-133] ✅ Create JIRA.md task tracker</li>
        <li>[TASK-134] ⚪ Review and finalize web app architecture</li>
        <li>[TASK-135] ⚪ Set up development environment</li>
        <li>[TASK-136] ✅ Initialize Next.js 16 project with TypeScript (2 points)</li>
        <li>[TASK-137] 🟡 Set up basic project structure and routing (2 points)</li>
        <li>[TASK-138] 🟡 Create homepage with links to existing content (1 point)</li>
        <li>[TASK-139] 🟡 Add documentation pages (README, Guidelines) (1 point)</li>
        <li>[TASK-140] 🟡 Configure Vercel deployment settings (1 point)</li>
        <li>[TASK-141] 🟡 Deploy initial version to Vercel free tier (1 point)</li>
        <li>[TASK-142] 🟡 Set up Playwright for UI testing (2 points)</li>
        <li>[TASK-143] ⚪ Extract Compass template from zip file</li>
        <li>[TASK-144] ⚪ Create detailed technical specification document</li>
        <li>[TASK-145] ⚪ Plan database schema</li>
        <li>[TASK-146] ⚪ Define API endpoints</li>
        <li>[TASK-147] ⚪ Create component hierarchy diagram</li>
        <li>[TASK-148] ⚪ Write first tutorial in MDX format (proof of concept)</li>
      </ul>

      <p><strong>Acceptance Criteria:</strong></p>
      <ul>
        <li>All documentation complete and reviewed</li>
        <li>Architecture decisions documented</li>
        <li>Development environment ready</li>
        <li>Basic Next.js app deployed to Vercel</li>
        <li>Vercel deployment URL accessible</li>
        <li>Playwright can access and test the UI</li>
        <li>Homepage displays existing content links</li>
        <li>Compass template integrated and tested (deferred to Sprint 1)</li>
        <li>One complete tutorial in MDX format (deferred to Sprint 1)</li>
      </ul>

      <p><strong>Sprint Review Date:</strong> End of Week 2</p>

      <hr />

      <h2>📊 Metrics to Track</h2>

      <h3>Development Metrics</h3>
      <ul>
        <li>Story points completed per sprint</li>
        <li>Velocity (average points per sprint)</li>
        <li>Code coverage percentage</li>
        <li>Build time</li>
        <li>Bundle size</li>
      </ul>

      <h3>User Metrics (Post-Launch)</h3>
      <ul>
        <li>Daily/Monthly active users</li>
        <li>Tutorial completion rates</li>
        <li>Average time per tutorial</li>
        <li>Quiz accuracy (immediate and spaced)</li>
        <li>Retention (1 week, 1 month return rate)</li>
      </ul>

      <h3>Learning Outcomes</h3>
      <ul>
        <li>Quiz performance: immediate vs. 1 week vs. 1 month</li>
        <li>Transfer task success rate</li>
        <li>Self-reported confidence ratings</li>
        <li>Comparison: notebook learners vs. web app learners</li>
      </ul>

      <hr />

      <h2>🔄 Process</h2>

      <h3>Definition of Ready (DoR)</h3>
      <p>A story is ready for Sprint Planning when:</p>
      <ul>
        <li>Acceptance criteria defined</li>
        <li>Dependencies identified</li>
        <li>Story points estimated</li>
        <li>Technical approach discussed</li>
      </ul>

      <h3>Definition of Done (DoD)</h3>
      <p>A story is complete when:</p>
      <ul>
        <li>Code written and reviewed</li>
        <li>Tests passing</li>
        <li>Documentation updated</li>
        <li>Deployed to preview environment</li>
        <li>Acceptance criteria met</li>
        <li>Product owner approved</li>
      </ul>

      <h3>Story Point Scale</h3>
      <ul>
        <li><strong>1 point:</strong> &lt; 2 hours (trivial change)</li>
        <li><strong>2 points:</strong> 2-4 hours (small feature)</li>
        <li><strong>3 points:</strong> 4-8 hours (half day)</li>
        <li><strong>5 points:</strong> 1-2 days (medium feature)</li>
        <li><strong>8 points:</strong> 3-4 days (large feature)</li>
        <li><strong>13 points:</strong> 1 week (complex feature)</li>
        <li><strong>21 points:</strong> 2 weeks (epic-level work, should be broken down)</li>
      </ul>

      <hr />

      <h2>🎨 Design System</h2>

      <h3>Component Library (To Build)</h3>

      <h4>Tutorial Components:</h4>
      <ul>
        <li><code>TutorialLayout</code> - Main layout with TOC</li>
        <li><code>Section</code> - Section with anchor</li>
        <li><code>CodeBlock</code> - Syntax highlighted code</li>
        <li><code>ExecutableCode</code> - Code with run button</li>
        <li><code>QuickCheck</code> - Retrieval practice questions</li>
        <li><code>CompletionProblem</code> - Fill-in-the-blank code</li>
        <li><code>PredictionPrompt</code> - Predict before reveal</li>
        <li><code>Misconception</code> - Common pitfall callout</li>
        <li><code>Reflection</code> - Metacognitive checkpoint</li>
        <li><code>MathBlock</code> - LaTeX rendering</li>
        <li><code>Figure</code> - Image with caption</li>
        <li><code>Callout</code> - Info/warning/tip boxes</li>
      </ul>

      <h4>Navigation Components:</h4>
      <ul>
        <li><code>Sidebar</code> - Tutorial navigation</li>
        <li><code>Breadcrumbs</code> - Location indicator</li>
        <li><code>NextPrevious</code> - Tutorial navigation</li>
        <li><code>ProgressBar</code> - Tutorial progress</li>
      </ul>

      <h4>Interactive Components:</h4>
      <ul>
        <li><code>Quiz</code> - Multi-choice questions</li>
        <li><code>CodeEditor</code> - Editable code</li>
        <li><code>OutputConsole</code> - Code execution results</li>
        <li><code>Plot</code> - Matplotlib rendering</li>
        <li><code>ConfidenceRating</code> - 1-5 scale widget</li>
      </ul>

      <hr />

      <h2>📖 Resources</h2>

      <h3>Documentation</h3>
      <ul>
        <li><a href="https://nextjs.org/docs">Next.js 16 Docs</a> - Latest with Turbopack</li>
        <li><a href="https://react.dev/">React 19 Docs</a> - Latest React documentation</li>
        <li><a href="https://tailwindcss.com/docs">Tailwind CSS v4 Docs</a> - Latest Tailwind</li>
        <li><a href="https://mdxjs.com/">MDX Documentation</a></li>
        <li><a href="https://pyodide.org/en/stable/">Pyodide Docs</a></li>
        <li><a href="./LEARNING_SCIENCE_REVIEW.md">Learning Science Research</a></li>
      </ul>

      <h3>Design References</h3>
      <ul>
        <li>Tailwind Plus Compass Template (local zip)</li>
        <li><a href="https://tailwindcss.com/plus/templates/compass">Compass Demo</a></li>
        <li><a href="https://syntax.fm/">Syntax.fm</a> - Good learning platform UX</li>
        <li><a href="https://www.freecodecamp.org/">FreeCodeCamp</a> - Interactive tutorials</li>
        <li><a href="https://brilliant.org/">Brilliant.org</a> - Active learning approach</li>
      </ul>

      <hr />

      <h2>✅ Completed Work Summary</h2>

      <h3>Sprint -1: Documentation Phase (Complete)</h3>
      <ul>
        <li>✅ Created README.md</li>
        <li>✅ Created PROJECT_GUIDELINES.md</li>
        <li>✅ Conducted learning science review</li>
        <li>✅ Created IMPROVEMENT_GUIDE.md</li>
        <li>✅ Fixed all critical technical inaccuracies (7 issues)</li>
        <li>✅ Fixed all high-priority guideline adherence issues (5 issues)</li>
        <li>✅ Added random_state for reproducibility across all notebooks</li>
        <li>✅ Fixed section numbering inconsistencies</li>
      </ul>

      <p><strong>Total Story Points Completed:</strong> 55 points</p>

      <hr />

      <h2>🚀 Next Actions</h2>

      <h3>Immediate (This Week)</h3>
      <ol>
        <li>✅ Review and approve JIRA.md structure</li>
        <li>✅ Initialize Next.js 16 project with TypeScript & latest React 19</li>
        <li>✅ Set up basic project structure and routing</li>
        <li>✅ Create homepage with links to existing content</li>
        <li>🟡 Configure and deploy to Vercel free tier</li>
        <li>⚪ Set up Playwright for UI testing (after deployment)</li>
        <li>⚪ Extract Compass template from zip file (deferred to Sprint 1)</li>
      </ol>

      <h3>This Sprint (Week 1-2)</h3>
      <ol>
        <li>✅ Create comprehensive task tracker (JIRA.md)</li>
        <li>🟡 Deploy basic Next.js site to Vercel</li>
        <li>⚪ Extract and review Compass template</li>
        <li>⚪ Write technical specification</li>
        <li>⚪ Design database schema</li>
        <li>⚪ Create one proof-of-concept tutorial in MDX</li>
      </ol>

      <h3>Next Sprint (Week 3-4)</h3>
      <ol>
        <li>Integrate Compass template design</li>
        <li>Build core web app infrastructure</li>
        <li>Set up MDX tutorial system</li>
        <li>Integrate Pyodide for code execution</li>
        <li>Implement first interactive learning components</li>
      </ol>

      <hr />

      <h2>📞 Stakeholders</h2>

      <p>
        <strong>Project Owner:</strong> [Your Name]<br />
        <strong>Development Team:</strong> [Team Members]<br />
        <strong>Reviewers:</strong> [Code Reviewers]<br />
        <strong>Beta Testers:</strong> TBD
      </p>

      <hr />

      <h2>🏷️ Labels/Tags</h2>

      <h3>Priority:</h3>
      <ul>
        <li>🔴 Critical</li>
        <li>🟠 High</li>
        <li>🟡 Medium</li>
        <li>🟢 Low</li>
      </ul>

      <h3>Type:</h3>
      <ul>
        <li>Epic</li>
        <li>Story</li>
        <li>Task</li>
        <li>Bug</li>
      </ul>

      <h3>Status:</h3>
      <ul>
        <li>⚪ To Do</li>
        <li>🟡 In Progress</li>
        <li>✅ Done</li>
        <li>🔴 Blocked</li>
      </ul>

      <h3>Category:</h3>
      <ul>
        <li>Content</li>
        <li>Frontend</li>
        <li>Backend</li>
        <li>Infrastructure</li>
        <li>Design</li>
        <li>Documentation</li>
      </ul>

      <hr />

      <p>
        <strong>Last Updated:</strong> 2025-10-25<br />
        <strong>Next Review:</strong> End of Sprint 0 (Week 2)
      </p>
    </article>
  );
}
