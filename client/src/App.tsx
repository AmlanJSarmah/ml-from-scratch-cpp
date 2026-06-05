import * as React from "react"

import {
  Combobox,
  ComboboxContent,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox"
import { Button } from "@/components/ui/button"
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

export function App() {
  const [model, setModel] = React.useState<string | null>(null)
  const [dataset, setDataset] = React.useState<string | null>(null)
  const [benchmarkData, setBenchmarkData] = React.useState<unknown | null>(null)
  const [benchmarkError, setBenchmarkError] = React.useState<string | null>(null)
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
        `http://localhost:8080/benchmarks?model=${modelParam}&dataset=${datasetParam}`
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
      .filter((item): item is { metric: string; custom: number; sklearn: number } =>
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
        <div className="mx-auto grid w-full max-w-xl gap-4 sm:grid-cols-2">
          <div className="space-y-2 text-left">
            <label
              className="text-sm font-medium text-foreground"
              htmlFor="model-combobox"
            >
              Model
            </label>
            <Combobox
              value={model}
              onValueChange={(value) => {
                setModel(value)
                setBenchmarkData(null)
                setBenchmarkError(null)
              }}
            >
              <ComboboxInput
                id="model-combobox"
                placeholder="Select a model"
                className="w-full"
              />
              <ComboboxContent>
                <ComboboxList>
                  {MODEL_OPTIONS.map((option) => (
                    <ComboboxItem
                      key={option}
                      value={option}
                      disabled={
                        (isCaliforniaHousing && option !== "Linear Regression") ||
                        (isIrisSelected &&
                          (option === "Logistic Regression" ||
                            option === "Naive Bayes"))
                      }
                    >
                      {option}
                    </ComboboxItem>
                  ))}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </div>
          <div className="space-y-2 text-left">
            <label
              className="text-sm font-medium text-foreground"
              htmlFor="dataset-combobox"
            >
              Dataset
            </label>
            <Combobox
              value={dataset}
              onValueChange={(value) => {
                setDataset(value)
                setBenchmarkData(null)
                setBenchmarkError(null)
              }}
            >
              <ComboboxInput
                id="dataset-combobox"
                placeholder="Select a dataset"
                className="w-full"
              />
              <ComboboxContent>
                <ComboboxList>
                  {DATASET_OPTIONS.map((option) => (
                    <ComboboxItem
                      key={option}
                      value={option}
                      disabled={
                        (option === "California Housing" &&
                          !isLinearRegression &&
                          model !== null) ||
                        (option === "Iris" && isIrisIncompatibleModel)
                      }
                    >
                      {option}
                    </ComboboxItem>
                  ))}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </div>
        </div>
        <div className="flex justify-center">
          <Button
            onClick={handleRunBenchmark}
            disabled={!model || !dataset || isRunning}
          >
            {isRunning ? "Running..." : "Run benchmark"}
          </Button>
        </div>
        {!benchmarkError && chartData.length > 0 ? (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
            {isClassificationModel ? (
              <>
                {chartDataForMetrics.length > 0 ? (
                  <ChartContainer
                    className="h-72 w-full aspect-auto"
                    config={{
                      custom: { label: "Custom", color: "#3b82f6" },
                      sklearn: { label: "Sklearn", color: "#1d4ed8" },
                    }}
                  >
                    <BarChart data={chartDataForMetrics} margin={{ left: 8, right: 8 }}>
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
                      <Bar dataKey="sklearn" fill="var(--color-sklearn)" radius={4} />
                    </BarChart>
                  </ChartContainer>
                ) : null}
                {chartDataForConfusion.length > 0 ? (
                  <ChartContainer
                    className="h-72 w-full aspect-auto"
                    config={{
                      custom: { label: "Custom", color: "#3b82f6" },
                      sklearn: { label: "Sklearn", color: "#1d4ed8" },
                    }}
                  >
                    <BarChart data={chartDataForConfusion} margin={{ left: 8, right: 8 }}>
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
                      <Bar dataKey="sklearn" fill="var(--color-sklearn)" radius={4} />
                    </BarChart>
                  </ChartContainer>
                ) : null}
              </>
            ) : (
              <ChartContainer
                className="h-72 w-full aspect-auto"
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
                  <Bar dataKey="sklearn" fill="var(--color-sklearn)" radius={4} />
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
