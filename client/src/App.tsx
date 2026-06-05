export function App() {
  return (
    <div className="flex min-h-screen w-screen items-center justify-center px-6 text-center">
      <div className="space-y-3">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Select model and dataset to compare
        </h1>
        <p className="text-sm text-muted-foreground sm:text-base">
          Compare custom made C++ model's performance with sklearn on various
          datasets
        </p>
      </div>
    </div>
  )
}

export default App
