/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#08090B", 900: "#0C0E12", 850: "#111318",
          800: "#161920", 700: "#1E222B", 600: "#2A2F3A",
          500: "#3B4250", 400: "#5A6273", 300: "#8A93A6",
          200: "#B6BECD", 100: "#DDE2EA",
        },
        red: { DEFAULT: "#F04A5E", soft: "#FF7385", deep: "#7A1B2A", wash: "#2A1218" },
        blue: { DEFAULT: "#3DD8E8", soft: "#7BE9F4", deep: "#0E5C66", wash: "#0C2429" },
        amber: { DEFAULT: "#F2B33D", wash: "#2A2113" },
        green: { DEFAULT: "#3ED598", wash: "#0F2A22" },
        violet: { DEFAULT: "#9B8CFF", wash: "#1C1930" },
      },
      fontFamily: {
        sans: ["Inter var", "Inter", "ui-sans-serif", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "SF Mono", "Menlo", "Consolas", "Liberation Mono", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.02em" }],
      },
      animation: {
        "pulse-node": "pulseNode 1.4s ease-in-out infinite",
        "flow": "flow 1.2s linear infinite",
        "fade-up": "fadeUp .35s cubic-bezier(.2,.8,.2,1) both",
        "sweep": "sweep 2.4s ease-in-out infinite",
      },
      keyframes: {
        pulseNode: { "0%,100%": { opacity: "1" }, "50%": { opacity: ".45" } },
        flow: { to: { strokeDashoffset: "-12" } },
        fadeUp: { from: { opacity: "0", transform: "translateY(6px)" }, to: { opacity: "1", transform: "none" } },
        sweep: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(300%)" } },
      },
    },
  },
  plugins: [],
};
