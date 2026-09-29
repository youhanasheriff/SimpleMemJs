# Contributing to SimpleMem (TypeScript)

Thanks for your interest in improving SimpleMem. Bug reports, benchmarks and pull requests are welcome.

## Setup

```bash
bun install
```

## Checks to run before opening a PR

```bash
bun test              # unit tests (Bun runtime)
npm run test:node     # unit tests (Node.js via Vitest)
npm run lint
npm run typecheck
npm run build
```

CI runs the same checks on every pull request.

## Pull requests

- Keep changes focused; one topic per PR.
- Add or update tests for behavior changes.
- Update `CHANGELOG.md` for user-visible changes.
- Do not commit API keys or `.env` files.

## Reporting bugs

Open an issue with the runtime (Bun, Node or Deno) and version, a minimal reproduction, and the expected versus actual behavior. For security issues see [SECURITY.md](SECURITY.md).
