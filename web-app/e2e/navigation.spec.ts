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
    await expect(sidebar).toContainText('Documentation');
    await expect(sidebar).toContainText('Supervised Learning');
  });

  test('can toggle sidebar on desktop', async ({ page }) => {
    // Set viewport to desktop size (xl breakpoint is 1280px)
    await page.setViewportSize({ width: 1280, height: 720 });

    // Find the sidebar
    const sidebar = page.locator('aside').first();
    await expect(sidebar).toBeVisible();

    // Click the toggle button inside the sidebar
    const sidebarToggle = sidebar.locator('button').first();
    await sidebarToggle.click();

    // Sidebar should now be hidden
    await expect(sidebar).toBeHidden();

    // Toggle button should now appear in the page content area (not in the sidebar nav)
    // It's in the Navbar component with class "max-xl:hidden"
    const navbarToggle = page.locator('button[class*="max-xl:hidden"]').first();
    await expect(navbarToggle).toBeVisible();

    // Click the navbar toggle to reopen sidebar
    await navbarToggle.click();

    // Sidebar should be visible again
    await expect(sidebar).toBeVisible();
  });

  test('mobile navigation works correctly', async ({ page }) => {
    // Set viewport to mobile size
    await page.setViewportSize({ width: 375, height: 667 });

    // Desktop sidebar should be hidden on mobile
    const sidebar = page.locator('aside').first();
    await expect(sidebar).toBeHidden();

    // Mobile menu button should be visible (class "xl:hidden")
    const mobileMenuButton = page.locator('button[class*="xl:hidden"]').first();
    await expect(mobileMenuButton).toBeVisible();

    // Click mobile menu button
    await mobileMenuButton.click();

    // Wait for dialog animation
    await page.waitForTimeout(300);

    // The mobile dialog has role="dialog" with class "xl:hidden"
    // The DialogPanel inside contains the navigation
    // We need to find links specifically within the DialogPanel (not the desktop sidebar)
    const dialogPanel = page.locator('[role="dialog"][class*="xl:hidden"] [class*="isolate"]');

    // Check that the dialog panel contains navigation links
    const linksInDialog = dialogPanel.locator('a[href^="/docs"]');
    await expect(linksInDialog.first()).toBeVisible();

    // Verify we can see both Documentation and Supervised Learning sections
    await expect(dialogPanel).toContainText('Documentation');
    await expect(dialogPanel).toContainText('Supervised Learning');
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
