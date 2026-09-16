import Header from "@/components/ui/Header";

export default function Layout({ children }) {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      {/* PAGE CONTENT */}
      <main className="flex-1">
        {children}   {/* ← THIS MUST EXIST */}
      </main>
    </div>
  );
}
