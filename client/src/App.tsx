import * as React from "react"

import {
  Combobox,
  ComboboxContent,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox"

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

  const isLinearRegression = model === "Linear Regression"
  const isCaliforniaHousing = dataset === "California Housing"

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
            <Combobox value={model} onValueChange={(value) => setModel(value)}>
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
                        isCaliforniaHousing && option !== "Linear Regression"
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
              onValueChange={(value) => setDataset(value)}
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
                        option === "California Housing" &&
                        !isLinearRegression &&
                        model !== null
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
      </div>
    </div>
  )
}

export default App
