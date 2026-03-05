# MCP Server Setup for Wan2.1 PWA

## Overview
This guide shows how to connect Claude Desktop to your local Wan2.1 PWA project directory for persistent file access and seamless development workflow.

## What is MCP?

**Model Context Protocol (MCP)** allows Claude Desktop to directly access and manipulate files in your local project directory, eliminating the need to copy files back and forth.

### Benefits
- 🔄 **Direct File Access** - Claude can read and write files directly in your project
- 💾 **Persistent Changes** - All edits are saved to your local filesystem
- 🚀 **Faster Workflow** - No manual copying of code between sessions
- 🔒 **Secure** - Runs locally on your machine, you control access

---

## Prerequisites

- **Claude Desktop** installed ([download here](https://claude.ai/download))
- **Node.js 18+** installed
- **Your Wan2.1 PWA project** cloned locally

---

## Installation & Setup

### Step 1: Install MCP SDK

Choose one of these methods:

#### Global Installation (Recommended)
```bash
npm install -g @modelcontextprotocol/sdk
```

#### Or Use npx (No global install needed)
```bash
npx @modelcontextprotocol/sdk --version
```

---

### Step 2: Get Your Project Path

You need the **absolute path** to your wan-pwa directory.

#### macOS/Linux
```bash
cd /path/to/Wan2.1/wan-pwa
pwd
# Example output: /Users/yourname/projects/Wan2.1/wan-pwa
```

#### Windows
```powershell
cd C:\path\to\Wan2.1\wan-pwa
cd
# Example output: C:\Users\yourname\projects\Wan2.1\wan-pwa
```

**Copy this path** - you'll need it in the next step.

---

### Step 3: Configure Claude Desktop

#### macOS

1. **Open Claude Desktop Settings**
   - Click **Claude** in menu bar → **Settings** (or `Cmd + ,`)
   - Navigate to **Developer** tab
   - Click **Edit Config**

2. **Add MCP Server Configuration**

   Paste this JSON (replace `YOUR_ABSOLUTE_PATH`):

   ```json
   {
     "mcpServers": {
       "wan-pwa": {
         "command": "npx",
         "args": [
           "-y",
           "@modelcontextprotocol/server-filesystem",
           "YOUR_ABSOLUTE_PATH"
         ]
       }
     }
   }
   ```

   **Example (macOS):**
   ```json
   {
     "mcpServers": {
       "wan-pwa": {
         "command": "npx",
         "args": [
           "-y",
           "@modelcontextprotocol/server-filesystem",
           "/Users/johnsmith/projects/Wan2.1/wan-pwa"
         ]
       }
     }
   }
   ```

3. **Save and Restart**
   - Save the file (`Cmd + S`)
   - Quit Claude Desktop (`Cmd + Q`)
   - Reopen Claude Desktop

#### Windows

1. **Open Claude Desktop Settings**
   - Click **Settings** icon → **Developer** tab
   - Click **Edit Config**

2. **Add MCP Server Configuration**

   Paste this JSON (replace `YOUR_ABSOLUTE_PATH`):

   ```json
   {
     "mcpServers": {
       "wan-pwa": {
         "command": "npx",
         "args": [
           "-y",
           "@modelcontextprotocol/server-filesystem",
           "YOUR_ABSOLUTE_PATH"
         ]
       }
     }
   }
   ```

   **Example (Windows):**
   ```json
   {
     "mcpServers": {
       "wan-pwa": {
         "command": "npx",
         "args": [
           "-y",
           "@modelcontextprotocol/server-filesystem",
           "C:\\Users\\johnsmith\\projects\\Wan2.1\\wan-pwa"
         ]
       }
     }
   }
   ```

   **Note:** Use double backslashes (`\\`) in Windows paths.

3. **Save and Restart**
   - Save the file
   - Close and reopen Claude Desktop

#### Linux

1. **Open Claude Desktop Settings**
   - Open Settings → Developer tab
   - Click **Edit Config**

2. **Add MCP Server Configuration**

   Paste this JSON (replace `YOUR_ABSOLUTE_PATH`):

   ```json
   {
     "mcpServers": {
       "wan-pwa": {
         "command": "npx",
         "args": [
           "-y",
           "@modelcontextprotocol/server-filesystem",
           "YOUR_ABSOLUTE_PATH"
         ]
       }
     }
   }
   ```

   **Example (Linux):**
   ```json
   {
     "mcpServers": {
       "wan-pwa": {
         "command": "npx",
         "args": [
           "-y",
           "@modelcontextprotocol/server-filesystem",
           "/home/johnsmith/projects/Wan2.1/wan-pwa"
         ]
       }
     }
   }
   ```

3. **Save and Restart**
   - Save the file
   - Restart Claude Desktop

---

### Step 4: Verify Connection

Open a new conversation in Claude Desktop and ask:

```
Can you list the files in my wan-pwa project?
```

**Expected Response:**
```
wan-pwa/
├── apps/
│   ├── web/
│   └── api/
├── packages/
│   ├── db/
│   └── types/
├── README.md
├── SETUP.md
├── package.json
...
```

If you see the file structure, **you're connected!** 🎉

---

## Using MCP with Your Project

### Common Tasks

#### View a File
```
Show me the content of apps/web/src/app/page.tsx
```

#### Edit a File
```
Update the Button component in apps/web/src/components/ui/button.tsx to add a loading state
```

#### Create New Files
```
Create a new API endpoint for video analytics in apps/api/routes/analytics.py
```

#### Run Commands
```
Run npm install in the web app directory
```

#### Database Migrations
```
Create a new migration to add a 'favorites' table
```

### Example Workflow

1. **Ask for Code Review**
   ```
   Review the generation.py file and suggest improvements
   ```

2. **Request New Features**
   ```
   Add a retry button to failed generations in the history page
   ```

3. **Debug Issues**
   ```
   Why is the credit deduction not working? Check the database functions
   ```

4. **Refactor Code**
   ```
   Extract the image upload logic into a reusable hook
   ```

---

## Alternative: Claude Code CLI

For terminal-based workflows:

```bash
# Install CLI
npm install -g claude-code

# Navigate to project
cd /path/to/Wan2.1/wan-pwa

# Start session
claude-code

# In the CLI
> connect /path/to/Wan2.1/wan-pwa
> list files
> edit apps/web/src/app/page.tsx
```

---

## Troubleshooting

### Issue: "MCP server not found"

**Solution:**
1. Verify `@modelcontextprotocol/sdk` is installed:
   ```bash
   npm list -g @modelcontextprotocol/sdk
   ```

2. Check your config path is absolute (not relative):
   ```bash
   # ✅ Correct
   "/Users/john/projects/Wan2.1/wan-pwa"

   # ❌ Wrong
   "~/projects/Wan2.1/wan-pwa"
   "./wan-pwa"
   ```

3. Restart Claude Desktop completely

---

### Issue: "Permission denied"

**Solution:**
1. Check directory permissions:
   ```bash
   ls -la /path/to/Wan2.1/wan-pwa
   ```

2. Ensure you have read/write access:
   ```bash
   chmod -R u+rw /path/to/Wan2.1/wan-pwa
   ```

3. Don't run Claude Desktop with `sudo`

---

### Issue: "Files not updating"

**Solution:**
1. MCP servers don't auto-reload on file changes
2. Ask Claude to "refresh" or "reload" the file
3. Restart the MCP server:
   - Quit Claude Desktop
   - Reopen and start new conversation

---

### Issue: "command not found: npx"

**Solution:**
1. Install Node.js 18+ from [nodejs.org](https://nodejs.org)
2. Verify installation:
   ```bash
   node --version
   npm --version
   npx --version
   ```

3. Restart terminal and Claude Desktop

---

### Issue: Windows Path with Spaces

**Solution:**
Use double backslashes and quotes:

```json
{
  "mcpServers": {
    "wan-pwa": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\John Smith\\projects\\Wan2.1\\wan-pwa"
      ]
    }
  }
}
```

---

## Security Best Practices

### ✅ DO:
- Only grant access to your project directory
- Use `.gitignore` for secrets and credentials
- Keep `.env` files out of version control
- Review MCP config before saving

### ❌ DON'T:
- Grant access to root directory (`/` or `C:\`)
- Share your Claude Desktop config publicly
- Commit API keys or secrets to git
- Run with elevated permissions unnecessarily

---

## Advanced Configuration

### Multiple Projects

You can configure multiple MCP servers:

```json
{
  "mcpServers": {
    "wan-pwa": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/wan-pwa"]
    },
    "other-project": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/other-project"]
    }
  }
}
```

### Read-Only Access

For code review without edit permissions:

```json
{
  "mcpServers": {
    "wan-pwa-readonly": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/wan-pwa",
        "--readonly"
      ]
    }
  }
}
```

---

## Comparison: MCP vs Current Session

| Feature | Current Session | With MCP Server |
|---------|----------------|-----------------|
| File Access | Temporary | Persistent |
| Edits | Need manual copy | Direct to files |
| Git Integration | Manual commands | Direct access |
| Multiple Projects | One at a time | Multiple servers |
| Setup Time | None | 5 minutes |
| Best For | Quick tasks | Deep development |

---

## Next Steps

1. ✅ Complete MCP setup using steps above
2. 🧪 Test connection with simple file operations
3. 🚀 Start using Claude for development tasks
4. 📚 Explore [MCP Documentation](https://modelcontextprotocol.io/docs) for advanced features

---

## Resources

- **MCP Protocol Documentation**: https://modelcontextprotocol.io/docs
- **Claude Desktop Download**: https://claude.ai/download
- **Filesystem Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
- **Claude Code Guide**: https://docs.claude.com/en/docs/claude-code
- **Wan2.1 PWA Docs**: See README.md, SETUP.md, DEPLOYMENT.md

---

## Support

### Getting Help

1. Check this documentation
2. Review troubleshooting section
3. Check [MCP GitHub Issues](https://github.com/modelcontextprotocol/typescript-sdk/issues)
4. Ask in Claude Desktop (once connected!)

### Common Questions

**Q: Do I need MCP for development?**
A: No, but it significantly improves the workflow for active development.

**Q: Does MCP work with claude.ai (web)?**
A: No, MCP is currently desktop-only. Use file uploads for web.

**Q: Can I use MCP with VS Code?**
A: MCP is for Claude Desktop. For VS Code, use the Claude Code extension.

**Q: Is my code sent to Anthropic?**
A: Only the files you discuss in conversations. MCP runs locally.

**Q: Can multiple people share an MCP config?**
A: Yes, but each person needs their own absolute path configured.

---

**Setup Complete?** Start building with Phase 4 features! 🎉
