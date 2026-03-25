import { AudioInputCard } from "../components/AudioInputCard";
import { TextInputCard } from "../components/TextInputCard";
import { ActionBar } from "../components/ActionBar";
import { ResultsPanel } from "../components/ResultsPanel";

export function Dashboard() {
  return (
    <div className="flex flex-col gap-5 w-full max-w-2xl mx-auto">
      <div className="flex flex-col gap-5">
        <AudioInputCard />
        <TextInputCard />
      </div>

      <ActionBar />

      <ResultsPanel />
    </div>
  );
}
