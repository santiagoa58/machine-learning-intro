import { promises as fs } from 'fs';
import path from 'path';
import matter from 'gray-matter';

export interface AlgorithmContent {
  id: string;
  content: string;
  metadata: {
    title: string;
    description: string;
    prerequisites?: string[];
    learningOutcomes?: string[];
  };
}

/**
 * Get the content directory path
 */
function getContentPath(...segments: string[]): string {
  return path.join(process.cwd(), 'content', ...segments);
}

/**
 * Load algorithm content from markdown file
 * @param algorithmId - The algorithm ID (e.g., 'linear-regression')
 * @returns Algorithm content with metadata
 */
export async function getAlgorithmContent(algorithmId: string): Promise<AlgorithmContent | null> {
  try {
    const filePath = getContentPath('algorithms', `${algorithmId}.md`);
    const fileContent = await fs.readFile(filePath, 'utf-8');

    // Parse frontmatter and content
    const { data, content } = matter(fileContent);

    return {
      id: algorithmId,
      content,
      metadata: {
        title: data.title || '',
        description: data.description || '',
        prerequisites: data.prerequisites || [],
        learningOutcomes: data.learningOutcomes || [],
      },
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
    const filePath = getContentPath('algorithms', `${algorithmId}.md`);
    await fs.access(filePath);
    return true;
  } catch {
    return false;
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
      .filter(file => file.endsWith('.md'))
      .map(file => file.replace('.md', ''));
  } catch {
    return [];
  }
}
