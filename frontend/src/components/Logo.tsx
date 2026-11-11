import markSvg from '../assets/icons/mark.svg'

interface LogoProps {
  className?: string
  onClick?: () => void
}

export function Logo({ className = '', onClick }: LogoProps) {
  return (
    <div
      className={`logo-brand ${className} ${onClick ? 'clickable' : ''}`}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <img src={markSvg} alt="SceneTIQ" className="logo-mark" width={28} height={28} />
      <span className="logo-text">scenetiq</span>
    </div>
  )
}
