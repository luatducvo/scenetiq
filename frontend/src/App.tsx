import { useCallback, useState } from 'react'
import { JobProgress } from './components/JobProgress'
import { SceneList } from './components/SceneList'
import { SceneViewer } from './components/SceneViewer'
import { UploadForm } from './components/UploadForm'
import { backendConfig as config } from './config'

type Screen =
  | { kind: 'scenes' }
  | { kind: 'upload' }
  | { kind: 'job'; jobId: string }
  | { kind: 'scene'; sceneId: string }

export default function App() {
  const [screen, setScreen] = useState<Screen>({ kind: 'scenes' })

  const openScene = useCallback((sceneId: string) => setScreen({ kind: 'scene', sceneId }), [])
  const openJob = useCallback((jobId: string) => setScreen({ kind: 'job', jobId }), [])

  if (!config.url || !config.token) {
    return (
      <div className="app">
        <div className="panel">
          <h2>Backend not configured</h2>
          <p>
            Set <code>VITE_BACKEND_URL</code> and <code>VITE_BACKEND_TOKEN</code> in{' '}
            <code>frontend/.env</code>, then restart <code>npm run dev</code>.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>SceneTIQ</h1>
        <nav>
          <button onClick={() => setScreen({ kind: 'scenes' })}>Scenes</button>
          <button onClick={() => setScreen({ kind: 'upload' })}>New Scene</button>
        </nav>
      </header>
      <main>
        {screen.kind === 'scenes' && (
          <SceneList config={config} onOpen={openScene} onOpenJob={openJob} />
        )}
        {screen.kind === 'upload' && <UploadForm config={config} onStarted={openJob} />}
        {screen.kind === 'job' && (
          <JobProgress config={config} jobId={screen.jobId} onSucceeded={openScene} />
        )}
        {screen.kind === 'scene' && <SceneViewer config={config} sceneId={screen.sceneId} />}
      </main>
    </div>
  )
}
