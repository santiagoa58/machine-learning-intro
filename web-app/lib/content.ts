import { promises as fs } from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { generateSummary, extractExcerpt } from './summarize';

export interface AlgorithmContent {
  id: string;
  content: string;
  summary: string | null;
  metadata: {
    title: string;
    description: string;
    prerequisites?: string[];
    learningOutcomes?: string[];
  };
  isMdx: boolean;
}

/**
 * Get the content directory path
 */
function getContentPath(...segments: string[]): string {
  return path.join(process.cwd(), 'content', ...segments);
}

/**
 * Load algorithm content from markdown or MDX file
 * @param algorithmId - The algorithm ID (e.g., 'linear-regression')
 * @returns Algorithm content with metadata
 */
export async function getAlgorithmContent(algorithmId: string): Promise<AlgorithmContent | null> {
  try {
    // Try MDX first, then fall back to MD
    let filePath = getContentPath('algorithms', `${algorithmId}.mdx`);
    let isMdx = true;

    try {
      await fs.access(filePath);
    } catch {
      // MDX doesn't exist, try MD
      filePath = getContentPath('algorithms', `${algorithmId}.md`);
      isMdx = false;
    }

    const fileContent = await fs.readFile(filePath, 'utf-8');

    // Parse frontmatter and content
    const { data, content } = matter(fileContent);

    // Generate summary server-side (fast excerpt extraction, cached)
    const { summary } = await generateSummary(content);

    return {
      id: algorithmId,
      content,
      summary, // Simple excerpt, no AI blocking
      metadata: {
        title: data.title || '',
        description: data.description || '',
        prerequisites: data.prerequisites || [],
        learningOutcomes: data.learningOutcomes || [],
      },
      isMdx,
    };
  } catch (error) {
    // If file doesn't exist, return null
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

/**
 * Check if content exists for an algorithm
 * @param algorithmId - The algorithm ID
 * @returns true if content exists
 */
export async function hasAlgorithmContent(algorithmId: string): Promise<boolean> {
  try {
    // Try MDX first
    let filePath = getContentPath('algorithms', `${algorithmId}.mdx`);
    await fs.access(filePath);
    return true;
  } catch {
    // Try MD
    try {
      const filePath = getContentPath('algorithms', `${algorithmId}.md`);
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * List all available algorithm content files
 * @returns Array of algorithm IDs that have content
 */
export async function listAlgorithmContent(): Promise<string[]> {
  try {
    const algorithmsDir = getContentPath('algorithms');
    const files = await fs.readdir(algorithmsDir);

    return files
      .filter(file => file.endsWith('.md') || file.endsWith('.mdx'))
      .map(file => file.replace(/\.(md|mdx)$/, ''));
  } catch {
    return [];
  }
}
