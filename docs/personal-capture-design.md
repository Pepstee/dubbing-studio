# Personal Capture design system — provisional

The operator will often review recordings from a phone, so the interface is a restrained,
calm evidence surface rather than an analytics dashboard.

- Register: product UI; dense when evidence requires it.
- Surface: neutral near-white with no decorative texture.
- Seed/action color: moss `oklch(0.400 0.106 150)`.
- Typography: system sans, fixed rem scale, readable at mobile widths.
- Structure: queue table followed by a linear segment editor; no generic card grid.
- Accessibility: WCAG AA contrast, visible focus, semantic labels, native controls, reduced
  motion, and no information encoded by color alone.
- Safety language distinguishes review, approval, rejection, and GIGA outbox delivery.

These choices are assumptions until tested with real recordings and operator feedback.
