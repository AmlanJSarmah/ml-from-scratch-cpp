import * as React from "react"
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"

const MODEL_OPTIONS = [
  "Linear Regression",
  "Logistic Regression",
  "Naive Bayes",
  "K Means",
] as const

const DATASET_OPTIONS = [
  "California Housing",
  "Breast Cancer",
  "Titanic survived",
  "Iris",
] as const

// Lightweight in-file replacements for the shadcn Combobox and Button used
// by this app. They intentionally keep behaviour identical for selection
// and disabled states but avoid relying on external UI primitives which
// weren't rendering correctly in the environment.

function CustomButton({
  children,
  onClick,
  disabled,
}: {
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 disabled:pointer-events-none bg-primary text-white`}
    >
      {children}
    </button>
  )
}

function CustomSelect({
  id,
  value,
  disabled,
  onChange,
  placeholder,
  className,
  options,
}: {
  id?: string
  value: string | null
  disabled?: boolean
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  options: string[]
}) {
  const [open, setOpen] = React.useState(false)
  const ref = React.useRef<HTMLDivElement | null>(null)

  React.useEffect(() => {
    function onDocumentClick(e: MouseEvent) {
      if (!ref.current) return
      if (!ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", onDocumentClick)
    return () => document.removeEventListener("mousedown", onDocumentClick)
  }, [])

  return (
    <div className={`relative ${className ?? ""}`} ref={ref}>
      <button
        id={id}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        className={`w-full text-left rounded-md border px-3 py-2 shadow-sm focus:ring-2 focus:ring-ring disabled:opacity-50 disabled:pointer-events-none bg-background`}
      >
        <span className={`${value ? "text-foreground" : "text-muted-foreground"}`}>
          {value ?? placeholder ?? "Select..."}
        </span>
      </button>

      {open && (
        <ul
          role="listbox"
          aria-activedescendant={value ?? undefined}
          className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md border bg-popover p-1 text-sm shadow-lg"
        >
          {options.length === 0 ? (
            <li className="px-2 py-1 text-muted-foreground">No options</li>
          ) : (
            options.map((option) => (
              <li
                key={option}
                role="option"
                aria-selected={option === value}
                onClick={() => {
                  onChange(option)
                  setOpen(false)
                }}
                className={`cursor-pointer rounded px-2 py-1 hover:bg-accent/40 ${option === value ? "bg-accent/60 font-semibold" : ""}`}
              >
                {option}
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  )
}

export function App() {
  const [model, setModel] = React.useState<string | null>(null)
  const [dataset, setDataset] = React.useState<string | null>(null)
  const [benchmarkData, setBenchmarkData] = React.useState<unknown | null>(null)
  const [benchmarkError, setBenchmarkError] = React.useState<string | null>(
    null
  )
  const [isRunning, setIsRunning] = React.useState(false)

  const isLinearRegression = model === "Linear Regression"
  const isCaliforniaHousing = dataset === "California Housing"
  const isIrisSelected = dataset === "Iris"
  const isIrisIncompatibleModel =
    model === "Naive Bayes" || model === "Logistic Regression"
  const isClassificationModel =
    model === "Naive Bayes" || model === "Logistic Regression"

  const toSnakeCase = (value: string) =>
    value.trim().toLowerCase().replace(/\s+/g, "_")

  const formatMetricLabel = (metric: string) => {
    if (metric.length <= 3) {
      return metric.toUpperCase()
    }
    return metric
      .replace(/_/g, " ")
      .replace(/\b\w/g, (match) => match.toUpperCase())
  }

  const handleRunBenchmark = async () => {
    if (!model || !dataset) {
      return
    }

    setIsRunning(true)
    setBenchmarkError(null)
    setBenchmarkData(null)
    const modelParam = toSnakeCase(model)
    const datasetParam = toSnakeCase(dataset)
    try {
      const response = await fetch(
        `https://ml-from-scratch-cpp.onrender.com/benchmarks?model=${modelParam}&dataset=${datasetParam}`
      )
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }
      const data = await response.json()
      setBenchmarkData(data)
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error occurred"
      setBenchmarkError(message)
      window.alert(message)
    } finally {
      setIsRunning(false)
    }
  }

  const chartData = React.useMemo(() => {
    if (!benchmarkData || typeof benchmarkData !== "object") {
      return []
    }

    const data = benchmarkData as {
      custom?: Record<string, unknown>
      sklearn?: Record<string, unknown>
    }

    if (!data.custom || !data.sklearn) {
      return []
    }

    const metrics = new Set([
      ...Object.keys(data.custom),
      ...Object.keys(data.sklearn),
    ])

    const toNumber = (value: unknown) => {
      if (typeof value === "number") {
        return value
      }
      if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value)
        return Number.isFinite(parsed) ? parsed : null
      }
      return null
    }

    return Array.from(metrics)
      .map((metric) => {
        const customValue = toNumber(data.custom?.[metric])
        const sklearnValue = toNumber(data.sklearn?.[metric])

        if (customValue === null || sklearnValue === null) {
          return null
        }

        return {
          metric,
          custom: customValue,
          sklearn: sklearnValue,
        }
      })
      .filter(
        (item): item is { metric: string; custom: number; sklearn: number } =>
          Boolean(item)
      )
  }, [benchmarkData])

  const chartDataForMetrics = React.useMemo(() => {
    const keySet = new Set(["accuracy", "precision", "recall", "f1"])
    return chartData.filter((item) => keySet.has(item.metric))
  }, [chartData])

  const chartDataForConfusion = React.useMemo(() => {
    const keySet = new Set(["tp", "fp", "tn", "fn"])
    return chartData.filter((item) => keySet.has(item.metric))
  }, [chartData])

  return (
    <div className="flex min-h-screen w-screen items-center justify-center px-6 text-center">
      <div className="space-y-6">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Select model and dataset to compare
        </h1>
        <p className="text-sm text-muted-foreground sm:text-base">
          Compare custom made C++ model's performance with sklearn on various
          datasets
        </p>
        <p className="text-sm text-muted-foreground sm:text-base">
          For source code of the custom C++ models and setup instructions visit{" "}
          <a
            className="font-medium text-foreground underline"
            href="https://github.com/AmlanJSarmah/ml-from-scratch-cpp"
            rel="noreferrer"
            target="_blank"
          >
            https://github.com/AmlanJSarmah/ml-from-scratch-cpp
          </a>
          .
        </p>
        <div
          role="alert"
          className="rounded-md border border-yellow-200 bg-yellow-50 px-4 py-2 text-sm text-yellow-800"
        >
          The server goes to sleep because of inactivity and takes a minute to
          restart and another minute or so to run the custom model and
          benchmark scripts. Please be patient
        </div>
        <div className="mx-auto grid w-full max-w-xl gap-4 sm:grid-cols-2">
          <div className="space-y-2 text-left">
            <label
              className="text-sm font-medium text-foreground"
              htmlFor="model-combobox"
            >
              Model
            </label>
            <CustomSelect
              id="model-combobox"
              value={model}
              disabled={isRunning}
              onChange={(value) => {
                setModel(value)
                setBenchmarkData(null)
                setBenchmarkError(null)
              }}
              placeholder="Select a model"
              className="w-full"
              options={MODEL_OPTIONS.filter((option) => {
                if (isCaliforniaHousing) {
                  return option === "Linear Regression"
                }
                if (isIrisSelected) {
                  return (
                    option !== "Logistic Regression" &&
                    option !== "Naive Bayes"
                  )
                }
                return true
              })}
            />
          </div>
          <div className="space-y-2 text-left">
            <label
              className="text-sm font-medium text-foreground"
              htmlFor="dataset-combobox"
            >
              Dataset
            </label>
            <CustomSelect
              id="dataset-combobox"
              value={dataset}
              disabled={isRunning}
              onChange={(value) => {
                setDataset(value)
                setBenchmarkData(null)
                setBenchmarkError(null)
              }}
              placeholder="Select a dataset"
              className="w-full"
              options={DATASET_OPTIONS.filter((option) => {
                if (option === "California Housing") {
                  return isLinearRegression || model === null
                }
                if (option === "Iris") {
                  return !isIrisIncompatibleModel
                }
                return true
              })}
            />
          </div>
        </div>
        <div className="flex justify-center">
          <CustomButton
            onClick={handleRunBenchmark}
            disabled={!model || !dataset || isRunning}
          >
            {isRunning ? "Running..." : "Run benchmark"}
          </CustomButton>
        </div>
        {!benchmarkError && chartData.length > 0 ? (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
            {isClassificationModel ? (
              <>
                {chartDataForMetrics.length > 0 ? (
                  <ChartContainer
                    className="aspect-auto h-72 w-full"
                    config={{
                      custom: { label: "Custom", color: "#3b82f6" },
                      sklearn: { label: "Sklearn", color: "#1d4ed8" },
                    }}
                  >
                    <BarChart
                      data={chartDataForMetrics}
                      margin={{ left: 8, right: 8 }}
                    >
                      <CartesianGrid vertical={false} />
                      <XAxis
                        dataKey="metric"
                        tickFormatter={formatMetricLabel}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        tickLine={false}
                        axisLine={false}
                        width={40}
                        tickFormatter={(value) =>
                          typeof value === "number"
                            ? value.toLocaleString(undefined, {
                                maximumFractionDigits: 3,
                              })
                            : String(value)
                        }
                      />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <ChartLegend content={<ChartLegendContent />} />
                      <Bar
                        dataKey="custom"
                        fill="var(--color-custom)"
                        radius={4}
                      />
                      <Bar
                        dataKey="sklearn"
                        fill="var(--color-sklearn)"
                        radius={4}
                      />
                    </BarChart>
                  </ChartContainer>
                ) : null}
                {chartDataForConfusion.length > 0 ? (
                  <ChartContainer
                    className="aspect-auto h-72 w-full"
                    config={{
                      custom: { label: "Custom", color: "#3b82f6" },
                      sklearn: { label: "Sklearn", color: "#1d4ed8" },
                    }}
                  >
                    <BarChart
                      data={chartDataForConfusion}
                      margin={{ left: 8, right: 8 }}
                    >
                      <CartesianGrid vertical={false} />
                      <XAxis
                        dataKey="metric"
                        tickFormatter={formatMetricLabel}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        tickLine={false}
                        axisLine={false}
                        width={40}
                        tickFormatter={(value) =>
                          typeof value === "number"
                            ? value.toLocaleString(undefined, {
                                maximumFractionDigits: 3,
                              })
                            : String(value)
                        }
                      />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <ChartLegend content={<ChartLegendContent />} />
                      <Bar
                        dataKey="custom"
                        fill="var(--color-custom)"
                        radius={4}
                      />
                      <Bar
                        dataKey="sklearn"
                        fill="var(--color-sklearn)"
                        radius={4}
                      />
                    </BarChart>
                  </ChartContainer>
                ) : null}
              </>
            ) : (
              <ChartContainer
                className="aspect-auto h-72 w-full"
                config={{
                  custom: { label: "Custom", color: "#3b82f6" },
                  sklearn: { label: "Sklearn", color: "#1d4ed8" },
                }}
              >
                <BarChart data={chartData} margin={{ left: 8, right: 8 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis
                    dataKey="metric"
                    tickFormatter={formatMetricLabel}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    width={40}
                    tickFormatter={(value) =>
                      typeof value === "number"
                        ? value.toLocaleString(undefined, {
                            maximumFractionDigits: 3,
                          })
                        : String(value)
                    }
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Bar dataKey="custom" fill="var(--color-custom)" radius={4} />
                  <Bar
                    dataKey="sklearn"
                    fill="var(--color-sklearn)"
                    radius={4}
                  />
                </BarChart>
              </ChartContainer>
            )}
          </div>
        ) : null}
        {benchmarkData || benchmarkError ? (
          <pre className="mx-auto w-full max-w-xl rounded-lg border bg-muted/40 p-4 text-left text-xs text-muted-foreground">
            {benchmarkError
              ? benchmarkError
              : JSON.stringify(benchmarkData, null, 2)}
          </pre>
        ) : null}
      </div>
    </div>
  )
}

export default App
