import { useCallback, useState } from 'react'
import { BackendStatus } from './components/BackendStatus'
import { JobProgress } from './components/JobProgress'
import { Logo } from './components/Logo'
import { SceneList } from './components/SceneList'
import { SceneViewer } from './components/SceneViewer'
import { UploadForm } from './components/UploadForm'
import { backendConfig as config } from './config'
import iconPlusSvg from './assets/icons/icon-plus.svg'

type Screen =
  | { kind: 'scenes' }
  | { kind: 'upload' }
  | { kind: 'job'; jobId: string }
  | { kind: 'scene'; sceneId: string }

export default function App() {
  const [screen, setScreen] = useState<Screen>({ kind: 'scenes' })

  const openScene = useCallback((sceneId: string) => setScreen({ kind: 'scene', sceneId }), [])
  const openJob = useCallback((jobId: string) => setScreen({ kind: 'job', jobId }), [])
  const openScenes = useCallback(() => setScreen({ kind: 'scenes' }), [])
  const openUpload = useCallback(() => setScreen({ kind: 'upload' }), [])

  if (!config.url || !config.token) {
    return (
      <div className="app-shell">
        <header className="app-bar">
          <Logo />
        </header>
        <main className="app-main">
          <div className="config-warning-panel">
            <h2>Backend not configured</h2>
            <p>
              Set <code>VITE_BACKEND_URL</code> and <code>VITE_BACKEND_TOKEN</code> in{' '}
              <code>frontend/.env</code>, then restart <code>npm run dev</code>.
            </p>
          </div>
        </main>
      </div>
    )
  }

  if (screen.kind === 'scene') {
    return (
      <SceneViewer
        config={config}
        sceneId={screen.sceneId}
        onBack={openScenes}
      />
    )
  }

  return (
    <div className="app-shell">
      <header className="app-bar">
        <Logo onClick={openScenes} />
        <div className="app-bar-actions">
          <BackendStatus config={config} />
          {screen.kind === 'scenes' && (
            <button className="btn btn-primary app-bar-new-btn" onClick={openUpload}>
              <img src={iconPlusSvg} alt="" width={18} height={18} />
              <span>New scene</span>
            </button>
          )}
        </div>
      </header>
      <main className="app-main">
        {screen.kind === 'scenes' && (
          <SceneList
            config={config}
            onOpen={openScene}
            onOpenJob={openJob}
            onNewScene={openUpload}
          />
        )}
        {screen.kind === 'upload' && (
          <UploadForm
            config={config}
            onStarted={(jobId) => openJob(jobId)}
            onCancel={openScenes}
          />
        )}
        {screen.kind === 'job' && (
          <JobProgress
            config={config}
            jobId={screen.jobId}
            onSucceeded={openScene}
            onBack={openScenes}
          />
        )}
      </main>
    </div>
  )
}
