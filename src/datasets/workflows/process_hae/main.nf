workflow auto {
  findStates(params, meta.config)
    | meta.workflow.run(
      auto: [publish: "state"]
    )
}

workflow run_wf {
  take:
  input_ch

  main:
  output_ch = input_ch

    // copy id to the state
    | map{ id, state ->
      def new_state = state + [dataset_id: id]
      [id, new_state]
    }

    | hae.run(
      fromState: [
        "input",
        "dataset_id",
        "dataset_name",
        "dataset_url",
        "dataset_reference",
        "dataset_summary",
        "dataset_description",
        "dataset_organism",
      ],
      toState: ["output"]
    )

    | setState([output_dataset: "output"])

  emit:
  output_ch
}