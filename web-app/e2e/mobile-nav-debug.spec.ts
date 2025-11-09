import { test } from '@playwright/test';

test.describe('Mobile Navigation - Detailed Debug', () => {
  test('debug mobile navigation dialog visibility', async ({ page }) => {
    await page.goto('http://localhost:3001');

    // Set viewport to mobile size
    await page.setViewportSize({ width: 375, height: 667 });

    console.log('=== Starting mobile navigation test ===');

    // Log initial state
    const sidebar = page.locator('aside').first();
    const sidebarVisible = await sidebar.isVisible();
    console.log('Sidebar visible on mobile:', sidebarVisible);

    // Find mobile menu button
    const mobileMenuButton = page.locator('button[class*="xl:hidden"]').first();
    const buttonVisible = await mobileMenuButton.isVisible();
    console.log('Mobile menu button visible:', buttonVisible);

    if (buttonVisible) {
      const buttonHTML = await mobileMenuButton.evaluate(el => el.outerHTML);
      console.log('Button HTML:', buttonHTML);
    }

    // Click mobile menu button
    console.log('Clicking mobile menu button...');
    await mobileMenuButton.click();

    // Wait a bit for dialog to open
    await page.waitForTimeout(500);

    // Check for dialog elements
    console.log('\n=== Checking dialog elements ===');

    const dialog = page.locator('[role="dialog"]');
    const dialogCount = await dialog.count();
    console.log('Dialog count:', dialogCount);

    if (dialogCount > 0) {
      const dialogVisible = await dialog.first().isVisible();
      console.log('Dialog visible:', dialogVisible);

      const dialogHTML = await dialog.first().evaluate(el => {
        return {
          outerHTML: el.outerHTML.substring(0, 500),
          classList: Array.from(el.classList),
          dataOpen: el.getAttribute('data-open'),
          ariaModal: el.getAttribute('aria-modal'),
        };
      });
      console.log('Dialog info:', JSON.stringify(dialogHTML, null, 2));
    }

    // Check for DialogPanel
    const dialogPanel = page.locator('[class*="fixed"][class*="inset-y-0"][class*="left-0"]');
    const panelCount = await dialogPanel.count();
    console.log('\nDialogPanel count:', panelCount);

    if (panelCount > 0) {
      const panelVisible = await dialogPanel.first().isVisible();
      console.log('DialogPanel visible:', panelVisible);

      const panelInfo = await dialogPanel.first().evaluate(el => {
        const computed = window.getComputedStyle(el);
        return {
          classList: Array.from(el.classList),
          display: computed.display,
          visibility: computed.visibility,
          opacity: computed.opacity,
          transform: computed.transform,
          left: computed.left,
          width: computed.width,
        };
      });
      console.log('DialogPanel computed styles:', JSON.stringify(panelInfo, null, 2));
    }

    // Check for navigation links
    console.log('\n=== Checking navigation content ===');
    const navLinks = page.locator('a[href^="/docs"]');
    const linkCount = await navLinks.count();
    console.log('Navigation link count:', linkCount);

    for (let i = 0; i < Math.min(linkCount, 3); i++) {
      const link = navLinks.nth(i);
      const linkVisible = await link.isVisible();
      const href = await link.getAttribute('href');
      const text = await link.textContent();
      console.log(`Link ${i}: href="${href}" text="${text}" visible=${linkVisible}`);
    }

    // Take a screenshot
    await page.screenshot({ path: 'test-results/mobile-nav-debug.png', fullPage: true });
    console.log('\n=== Screenshot saved ===');
  });

  test('check if MobileNavigation component is rendered', async ({ page }) => {
    await page.goto('http://localhost:3001');
    await page.setViewportSize({ width: 375, height: 667 });

    // The MobileNavigation Dialog has className "xl:hidden"
    const mobileDialogContainer = page.locator('div[role="dialog"][class*="xl:hidden"]');
    const exists = await mobileDialogContainer.count() > 0;
    console.log('Mobile dialog container exists:', exists);

    // Click the button
    const button = page.locator('button[class*="xl:hidden"]').first();
    await button.click();
    await page.waitForTimeout(500);

    // Check if dialog backdrop appears
    const backdrop = page.locator('[class*="bg-gray-950/25"]');
    const backdropCount = await backdrop.count();
    console.log('Backdrop count:', backdropCount);

    if (backdropCount > 0) {
      const backdropVisible = await backdrop.first().isVisible();
      console.log('Backdrop visible:', backdropVisible);
    }
  });

  test('try to interact with navigation after dialog opens', async ({ page }) => {
    await page.goto('http://localhost:3001');
    await page.setViewportSize({ width: 375, height: 667 });

    // Click mobile menu
    await page.locator('button[class*="xl:hidden"]').first().click();
    await page.waitForTimeout(500);

    // Try to find links within the dialog panel specifically
    const dialogPanel = page.locator('div[class*="fixed inset-y-0 left-0"]');
    const linksInPanel = dialogPanel.locator('a');
    const linkCount = await linksInPanel.count();
    console.log('Links in dialog panel:', linkCount);

    // Try clicking a link if found
    if (linkCount > 0) {
      const firstLink = linksInPanel.first();
      const isVisible = await firstLink.isVisible();
      const href = await firstLink.getAttribute('href');
      console.log(`First link in panel: href="${href}" visible=${isVisible}`);

      if (isVisible) {
        await firstLink.click();
        console.log('Successfully clicked link in mobile nav');
      }
    }
  });
});