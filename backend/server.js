const express = require('express');
const cors = require('cors');
const { PythonShell } = require('python-shell');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

// Identify Python binary inside virtual environment
const isWindows = process.platform === 'win32';
const venvPythonPath = isWindows
  ? path.join(__dirname, 'venv', 'Scripts', 'python.exe')
  : path.join(__dirname, 'venv', 'bin', 'python');

app.post('/api/chat', (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: 'Question is required' });

  const options = {
    mode: 'text',
    pythonPath: venvPythonPath,
    pythonOptions: ['-u'],
    args: [question],
    env: { ...process.env }
  };

  PythonShell.run('rag_query.py', options)
    .then(results => {
      res.json({ answer: results.join('\n') });
    })
    .catch(err => {
      console.error(err);
      res.status(500).json({ error: 'RAG Pipeline execution failed.' });
    });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));