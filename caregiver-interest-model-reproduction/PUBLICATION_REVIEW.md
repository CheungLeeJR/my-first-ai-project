# Public release and data-governance notes

This repository is a sanitized code-and-aggregate-results release. Before publishing any additional research artifacts, verify that you have permission from the relevant data owner or supervisor.

## Safe to keep public in this portfolio

- source code written for the reproduction pipeline;
- unit tests built from synthetic/non-participant values;
- aggregate row counts and class counts;
- fold-level and summary model metrics;
- aggregate confusion matrices;
- a sanitized run configuration with no local/raw-data path.

## Keep out of the public repository unless explicitly authorized

- raw survey rows;
- participant-level rebuilt feature tables;
- row-level OOF predictions;
- names, contact details, exact addresses, free-text answers, or direct identifiers;
- confidential study identifiers, internal paths, credentials, or unpublished collaboration material.

The CLI therefore disables row-level artifacts by default and requires the explicit `--include-row-level-artifacts` flag to create them.
