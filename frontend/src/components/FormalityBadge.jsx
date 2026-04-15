/**
 * FormalityBadge Component
 *
 * Displays a color-coded badge for Korean formality levels
 */

const FORMALITY_INFO = {
  hasipsioche: {
    korean: '하십시오체',
    english: 'Formal Polite',
    className: 'badge-formal'
  },
  haeyoche: {
    korean: '해요체',
    english: 'Polite Informal',
    className: 'badge-polite'
  },
  haeche: {
    korean: '해체',
    english: 'Casual',
    className: 'badge-casual'
  }
};

export default function FormalityBadge({ level, showKorean = true }) {
  const info = FORMALITY_INFO[level];

  if (!info) return null;

  return (
    <span className={info.className}>
      {showKorean && `${info.korean} - `}{info.english}
    </span>
  );
}
