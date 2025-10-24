# Contributing to Wan2.1 PWA

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/wan-pwa.git
   cd wan-pwa
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Set up environment variables (see SETUP.md)

5. Start development:
   ```bash
   npm run dev
   ```

## Project Structure

```
wan-pwa/
├── apps/
│   ├── web/          # Next.js frontend
│   └── api/          # FastAPI backend
├── packages/
│   ├── ui/           # Shared UI components
│   ├── db/           # Database schema
│   └── types/        # TypeScript types
```

## Code Style

### TypeScript/JavaScript
- Use TypeScript for all new code
- Follow ESLint rules
- Use Prettier for formatting
- Prefer functional components with hooks

### Python
- Follow PEP 8 style guide
- Use type hints
- Document functions with docstrings
- Use async/await for async operations

### Formatting
```bash
npm run format  # Format all code
npm run lint    # Check linting
```

## Making Changes

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes
- Write clean, readable code
- Add comments for complex logic
- Update documentation as needed

### 3. Test Your Changes
```bash
npm run test       # Run tests
npm run build      # Verify build works
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add new feature"
```

#### Commit Message Format
Follow Conventional Commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

Examples:
```
feat: add video download button
fix: resolve credit deduction bug
docs: update setup instructions
```

### 5. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 6. Create Pull Request
- Go to the original repository
- Click "New Pull Request"
- Select your branch
- Fill out the PR template
- Submit for review

## Pull Request Guidelines

### PR Title
Use the same format as commit messages:
```
feat: add dark mode support
fix: resolve authentication issue
```

### PR Description
Include:
- What changes were made
- Why the changes were necessary
- How to test the changes
- Screenshots (if UI changes)
- Related issues (if applicable)

### Example PR Description
```markdown
## Changes
- Added dark mode toggle to settings
- Implemented theme persistence in localStorage
- Updated all components to support dark mode

## Why
Users requested dark mode for better viewing experience at night

## Testing
1. Click the theme toggle in settings
2. Verify colors change throughout the app
3. Refresh page and verify theme persists

## Screenshots
[Before/After screenshots]

Closes #123
```

## Feature Requests

### Before Submitting
- Check if feature already exists
- Search existing issues/PRs
- Consider if it fits project scope

### Creating Feature Request
1. Open new issue
2. Use "Feature Request" template
3. Describe:
   - The problem it solves
   - Proposed solution
   - Alternative solutions considered
   - Additional context

## Bug Reports

### Before Submitting
- Ensure you're on latest version
- Search existing issues
- Try to reproduce consistently

### Creating Bug Report
1. Open new issue
2. Use "Bug Report" template
3. Include:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots/logs
   - Environment details

## Areas to Contribute

### Frontend
- UI/UX improvements
- New prompt templates
- Performance optimizations
- Accessibility enhancements

### Backend
- API optimizations
- New generation features
- Background job processing
- Caching strategies

### Documentation
- Setup guides
- API documentation
- Code examples
- Tutorial videos

### Testing
- Unit tests
- Integration tests
- E2E tests
- Performance tests

## Code Review Process

1. **Automated Checks**
   - Linting
   - Type checking
   - Tests
   - Build verification

2. **Manual Review**
   - Code quality
   - Best practices
   - Documentation
   - Test coverage

3. **Feedback**
   - Address review comments
   - Make requested changes
   - Discuss disagreements respectfully

4. **Approval**
   - At least one approval required
   - All checks must pass
   - No merge conflicts

## Getting Help

- **Documentation**: Check SETUP.md and README.md
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions
- **Discord**: Join our community (if available)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in project documentation

Thank you for contributing! 🎉
