import { useEffect } from "react";
import { useStore } from "../../store/useStore";

/** The single autoplay ticker for the whole console.
 *
 *  Deliberately mounted once in the Shell rather than per page: the demo must
 *  keep advancing while the presenter navigates between Simulation, Evidence
 *  and Logs. Having each page own its own loop meant "Run Demo" did nothing
 *  unless you happened to be standing on the right screen.
 */
export function RunDriver() {
  const runState = useStore((s) => s.runState);
  const speed = useStore((s) => s.speed);
  const redControl = useStore((s) => s.redControl);
  const blueControl = useStore((s) => s.blueControl);
  const rev = useStore((s) => s.rev);
  const episode = useStore((s) => s.episode);
  const phase = episode?.phase ?? "idle";

  useEffect(() => {
    if (runState !== "running" || !episode) return;

    // Episode finished: roll the next one so a presenter can leave it running.
    // Not when a human is playing, though — rolling on a timer yanks the
    // scoreboard and the ground-truth reveal off screen before they can read
    // the outcome of the move they just made.
    if (episode.ended) {
      if (redControl === "human" || blueControl === "human") return;
      const id = setTimeout(() => {
        useStore.getState().newEpisode();
        useStore.getState().run();
      }, 1100 / speed);
      return () => clearTimeout(id);
    }

    // A human owns this turn — wait for them rather than stepping past it.
    const waitingOnHuman =
      (phase === "awaiting_red" && redControl === "human") ||
      (phase === "awaiting_blue" && blueControl === "human");
    if (waitingOnHuman) return;

    const id = setTimeout(() => useStore.getState().stepOnce(), 620 / speed);
    return () => clearTimeout(id);
  }, [runState, speed, redControl, blueControl, episode, phase, rev]);

  return null;
}
