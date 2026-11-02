import { useCallback, useState } from 'react'
import type { BackendConfig } from './api'
import { JobProgress } from './components/JobProgress'
import { SceneList } from './components/SceneList'
import { SceneViewer } from './components/SceneViewer'
import { SettingsPanel } from './components/SettingsPanel'
import { UploadForm } from './components/UploadForm'
import { loadConfig, saveConfig } from './settings'

type Screen =
  | { kind: 'settings' }
  | { kind: 'scenes' }
  | { kind: 'upload' }
  | { kind: 'job'; jobId: string }
  | { kind: 'scene'; sceneId: string }

export default function App() {
  const [config, setConfig] = useState<BackendConfig>(loadConfig)
  const [screen, setScreen] = useState<Screen>(() =>
    config.token ? { kind: 'scenes' } : { kind: 'settings' },
  )

  const openScene = useCallback((sceneId: string) => setScreen({ kind: 'scene', sceneId }), [])
  const openJob = useCallback((jobId: string) => setScreen({ kind: 'job', jobId }), [])

  return (
    <div className="app">
      <header>
        <h1>SceneTIQ</h1>
        <nav>
          <button onClick={() => setScreen({ kind: 'scenes' })}>Scenes</button>
          <button onClick={() => setScreen({ kind: 'upload' })}>New Scene</button>
          <button onClick={() => setScreen({ kind: 'settings' })}>Settings</button>
        </nav>
      </header>
      <main>
        {screen.kind === 'settings' && (
          <SettingsPanel
            config={config}
            onSave={(cfg) => {
              saveConfig(cfg)
              setConfig(cfg)
              setScreen({ kind: 'scenes' })
            }}
          />
        )}
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
