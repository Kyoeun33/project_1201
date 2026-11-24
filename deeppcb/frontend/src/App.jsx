import { useEffect, useState } from 'react'
import './App.css'

function UploadCard({ title, subtitle, file, onFile, kind }) {
  const [preview, setPreview] = useState('')

  useEffect(() => {
    if (!file) {
      setPreview('')
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  return (
    <label className={`upload-card ${file ? 'has-file' : ''}`} htmlFor={`file-${kind}`}>
      <input
        id={`file-${kind}`}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={(event) => onFile(event.target.files?.[0] ?? null)}
      />
      {preview ? (
        <div className="preview-wrap">
          <img src={preview} alt={`${title} 미리보기`} />
          <span className="change-image">이미지 변경</span>
        </div>
      ) : (
        <div className="upload-empty">
          <span className="upload-icon" aria-hidden="true">＋</span>
          <strong>이미지를 선택하거나 끌어다 놓으세요</strong>
          <span>PNG, JPG, WEBP · 최대 20MB</span>
        </div>
      )}
      <div className="upload-caption">
        <div><span className={`step-dot ${kind}`}>{kind === 'reference' ? '01' : '02'}</span><strong>{title}</strong></div>
        <span>{file ? file.name : subtitle}</span>
      </div>
    </label>
  )
}

function App() {
  const [reference, setReference] = useState(null)
  const [inspection, setInspection] = useState(null)
  const [model, setModel] = useState('fast')
  const [notice, setNotice] = useState('')

  const ready = reference && inspection

  function handleAnalyze() {
    setNotice('이미지 업로드는 준비됐어요. 실제 추론을 실행하려면 Python 분석 API 연결이 필요합니다.')
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="DeepPCB 홈">
          <span className="brand-mark"><span /></span>
          <span>DeepPCB<span className="brand-light"> Vision</span></span>
        </a>
        <nav className="top-nav" aria-label="주 메뉴">
          <a className="active" href="#inspection">검사하기</a>
          <a href="#workflow">워크플로</a>
        </nav>
        <div className="connection"><span className="status-dot" /> API 연결 대기</div>
      </header>

      <main id="top">
        <section className="intro" id="inspection">
          <div className="eyebrow"><span /> PCB QUALITY INSPECTION</div>
          <h1>두 이미지를 비교해<br /><em>불량을 찾아보세요.</em></h1>
          <p>정상 템플릿과 검사 이미지를 올리면, DeepPCB 모델이<br className="desktop-break" /> 기판의 차이를 분석할 준비를 합니다.</p>
        </section>

        <section className="workbench" aria-label="PCB 이미지 검사">
          <div className="section-head">
            <div><span className="section-kicker">새 검사</span><h2>검사 이미지 준비</h2></div>
            <span className="required-note"><span>*</span> 두 이미지 모두 필요합니다</span>
          </div>

          <div className="upload-grid">
            <UploadCard title="정상 템플릿" subtitle="비교 기준이 되는 정상 보드" file={reference} onFile={setReference} kind="reference" />
            <div className="compare-mark" aria-hidden="true"><span>＋</span></div>
            <UploadCard title="검사 이미지" subtitle="불량 여부를 확인할 보드" file={inspection} onFile={setInspection} kind="inspection" />
          </div>

          <div className="options-row">
            <div className="model-select-wrap">
              <label htmlFor="model">검사 모델</label>
              <select id="model" value={model} onChange={(event) => setModel(event.target.value)}>
                <option value="fast">속도형 모델 <span>빠른 후보 탐지</span></option>
                <option value="accurate">정밀형 모델 · Faster R-CNN</option>
              </select>
            </div>
            <button className="analyze-button" type="button" disabled={!ready} onClick={handleAnalyze}>
              <span>검사 시작</span><span className="button-arrow">↗</span>
            </button>
          </div>

          <div className={`api-note ${ready ? 'ready' : ''}`} role="status">
            <span className="info-icon">i</span>
            {notice || (ready ? '이미지가 준비됐습니다. 검사 실행은 Python 분석 API 연결 후 사용할 수 있습니다.' : '이미지 업로드 후 검사 준비 상태를 확인할 수 있습니다.')}
          </div>
        </section>

        <section className="workflow" id="workflow">
          <div className="workflow-title"><span className="section-kicker">간단한 검사 과정</span><h2>세 단계로 확인하세요</h2></div>
          <div className="workflow-grid">
            <article><span className="workflow-number">01</span><div><strong>이미지 업로드</strong><p>정상 보드와 검사 보드 이미지를 선택합니다.</p></div><span className="workflow-icon">▧</span></article>
            <article><span className="workflow-number">02</span><div><strong>모델 선택</strong><p>빠른 후보 탐지 또는 정밀 검사를 고릅니다.</p></div><span className="workflow-icon">⌘</span></article>
            <article><span className="workflow-number">03</span><div><strong>결과 확인</strong><p>불량 위치와 종류, 점수를 확인합니다.</p></div><span className="workflow-icon">↗</span></article>
          </div>
        </section>
      </main>

      <footer><span>DeepPCB Vision</span><span>Template-based PCB defect inspection</span></footer>
    </div>
  )
}

export default App
