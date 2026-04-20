include { findArgumentSchema } from "${meta.resources_dir}/helper.nf"

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

    // | check_dataset_with_schema.run(
    //   fromState: { id, state ->
    //     def schema = findArgumentSchema(meta.config, "input")
    //     def schemaYaml = tempFile("schema.yaml")
    //     writeYaml(schema, schemaYaml)
    //     [
    //       "input": state.input,
    //       "schema": schemaYaml
    //     ]
    //   },
    //   toState: { id, output, state ->
    //     // read the output to see if dataset passed the qc
    //     def checks = readYaml(output.output)
    //     state + [
    //       "dataset": checks["exit_code"] == 0 ? state.input : null,
    //     ]
    //   }
    // )

    // // remove datasets which didn't pass the schema check
    // | filter { id, state ->
    //   state.dataset != null
    // }

    | process_dataset.run(
      fromState: [
        input_sp: "input_sp",
        "input_sc": "input_sc"
      ],
      toState: [
        output_spatial_dataset: "output_spatial_dataset",
        output_scrnaseq_reference: "output_scrnaseq_reference"
      ]
    )

    // only output the files for which an output file was specified
    | setState(["output_spatial_dataset", "output_scrnaseq_reference"])

  emit:
  output_ch
}
