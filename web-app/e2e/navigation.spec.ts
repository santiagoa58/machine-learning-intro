import { test, expect } from '@playwright/test';

test.describe('Navigation and Sidebar', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3001');
  });

  test('homepage loads successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Machine Learning Introduction/);
    await expect(page.locator('h1')).toContainText('Machine Learning Introduction');
  });

  test('sidebar is visible on desktop by default', async ({ page }) => {
    // Set viewport to desktop size
    await page.setViewportSize({ width: 1280, height: 720 });

    // Sidebar should be visible
    const sidebar = page.locator('aside').first();
    await expect(sidebar).toBeVisible();

    // Should contain navigation links
    await expect(sidebar).toContainText('Reference');
    await expect(sidebar).toContainText('Supervised Learning');
  });

  test('documentation pages load successfully', async ({ page }) => {
    const docPages = [
      { path: '/docs/readme', h1Contains: 'Machine Learning Introduction' },
      { path: '/docs/guidelines', h1Contains: 'Project Guidelines' },
      { path: '/docs/learning-science', h1Contains: 'Learning Science Review' },
      { path: '/docs/improvement-guide', h1Contains: 'Improvement Guide' },
      { path: '/docs/jira', h1Contains: 'Task Tracker' },
    ];

    for (const doc of docPages) {
      await page.goto(`http://localhost:3001${doc.path}`);

      // Page should load successfully (not 404)
      await expect(page.locator('h1')).toBeVisible();
      await expect(page.locator('h1')).toContainText(doc.h1Contains);

      // Breadcrumbs should be present
      const breadcrumbs = page.locator('nav[aria-label="Breadcrumb"]');
      await expect(breadcrumbs).toBeVisible();
      await expect(breadcrumbs).toContainText('ML Learning Path');
      await expect(breadcrumbs).toContainText('Documentation');
    }
  });

  test('algorithm pages load successfully', async ({ page }) => {
    const algorithms = [
      { id: 'linear-regression', name: 'Linear Regression' },
      { id: 'knn', name: 'K-Nearest Neighbors' },
      { id: 'logistic-regression', name: 'Logistic Regression' },
      { id: 'svm', name: 'Support Vector Machines' },
    ];

    for (const algo of algorithms) {
      await page.goto(`http://localhost:3001/learn/${algo.id}`);

      // Page should load successfully (not 404)
      await expect(page.locator('h1')).toBeVisible();
      await expect(page.locator('h1')).toContainText(algo.name);

      // Should show "coming soon" message
      await expect(page.locator('text=Coming Soon')).toBeVisible();

      // Breadcrumbs should be present
      const breadcrumbs = page.locator('nav[aria-label="Breadcrumb"]');
      await expect(breadcrumbs).toBeVisible();
      await expect(breadcrumbs).toContainText('ML Learning Path');
      await expect(breadcrumbs).toContainText('Supervised Learning');
    }
  });

  test('sidebar navigation links work', async ({ page }) => {
    // Set viewport to desktop
    await page.setViewportSize({ width: 1280, height: 720 });

    // Click on a documentation link in sidebar
    const sidebar = page.locator('aside').first();
    await sidebar.locator('a[href="/docs/guidelines"]').click();

    // Should navigate to guidelines page
    await expect(page).toHaveURL(/\/docs\/guidelines/);
    await expect(page.locator('h1')).toContainText('Project Guidelines');

    // Click on an algorithm link in sidebar
    await sidebar.locator('a[href="/learn/linear-regression"]').click();

    // Should navigate to Linear Regression page
    await expect(page).toHaveURL(/\/learn\/linear-regression/);
    await expect(page.locator('h1')).toContainText('Linear Regression');
  });

  test('breadcrumb navigation works', async ({ page }) => {
    // Navigate to a doc page
    await page.goto('http://localhost:3001/docs/guidelines');

    // Click home breadcrumb
    const breadcrumbs = page.locator('nav[aria-label="Breadcrumb"]');
    await breadcrumbs.locator('a[href="/"]').click();

    // Should navigate to homepage
    await expect(page).toHaveURL('http://localhost:3001/');
    await expect(page.locator('h1')).toContainText('Machine Learning Introduction');
  });

  test('active page is highlighted in sidebar', async ({ page }) => {
    // Set viewport to desktop
    await page.setViewportSize({ width: 1280, height: 720 });

    // Navigate to guidelines page
    await page.goto('http://localhost:3001/docs/guidelines');

    // The guidelines link should have aria-current="page"
    const sidebar = page.locator('aside').first();
    const guidelinesLink = sidebar.locator('a[href="/docs/guidelines"]');
    await expect(guidelinesLink).toHaveAttribute('aria-current', 'page');

    // Navigate to another page
    await page.goto('http://localhost:3001/docs/readme');

    // The readme link should now be active
    const readmeLink = sidebar.locator('a[href="/docs/readme"]');
    await expect(readmeLink).toHaveAttribute('aria-current', 'page');

    // Guidelines link should no longer be active
    await expect(guidelinesLink).not.toHaveAttribute('aria-current', 'page');
  });
});
