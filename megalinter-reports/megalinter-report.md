## ✅[MegaLinter](https://megalinter.io/9.5.0) analysis: Success



| Descriptor  |                                               Linter                                                |Files|Fixed|Errors|Warnings|Elapsed time|
|-------------|-----------------------------------------------------------------------------------------------------|----:|----:|-----:|-------:|-----------:|
|✅ JSON      |[prettier](https://megalinter.io/9.5.0/descriptors/json_prettier)                                    |    1|    0|     0|       0|       0.33s|
|✅ JSON      |[v8r](https://megalinter.io/9.5.0/descriptors/json_v8r)                                              |    1|     |     0|       0|       2.44s|
|✅ MARKDOWN  |[markdownlint](https://megalinter.io/9.5.0/descriptors/markdown_markdownlint)                        |    2|    0|     0|       0|       0.75s|
|✅ MARKDOWN  |[markdown-table-formatter](https://megalinter.io/9.5.0/descriptors/markdown_markdown_table_formatter)|    2|    0|     0|       0|       0.25s|
|✅ PYTHON    |[ruff](https://megalinter.io/9.5.0/descriptors/python_ruff)                                          |   71|    1|     0|       0|       0.29s|
|✅ REPOSITORY|[checkov](https://megalinter.io/9.5.0/descriptors/repository_checkov)                                |  yes|     |    no|      no|      21.51s|
|✅ REPOSITORY|[gitleaks](https://megalinter.io/9.5.0/descriptors/repository_gitleaks)                              |  yes|     |    no|      no|       1.05s|
|✅ REPOSITORY|[git_diff](https://megalinter.io/9.5.0/descriptors/repository_git_diff)                              |  yes|     |    no|      no|       0.02s|
|✅ REPOSITORY|[osv-scanner](https://megalinter.io/9.5.0/descriptors/repository_osv_scanner)                        |  yes|     |    no|      no|       0.47s|
|✅ REPOSITORY|[secretlint](https://megalinter.io/9.5.0/descriptors/repository_secretlint)                          |  yes|     |    no|      no|       0.95s|
|✅ REPOSITORY|[syft](https://megalinter.io/9.5.0/descriptors/repository_syft)                                      |  yes|     |    no|      no|       2.11s|
|✅ REPOSITORY|[trivy-sbom](https://megalinter.io/9.5.0/descriptors/repository_trivy_sbom)                          |  yes|     |    no|      no|       0.74s|
|✅ REPOSITORY|[trufflehog](https://megalinter.io/9.5.0/descriptors/repository_trufflehog)                          |  yes|     |    no|      no|       4.26s|
|✅ YAML      |[prettier](https://megalinter.io/9.5.0/descriptors/yaml_prettier)                                    |    5|    0|     0|       0|       0.55s|
|✅ YAML      |[v8r](https://megalinter.io/9.5.0/descriptors/yaml_v8r)                                              |    5|     |     0|       0|       5.27s|
|✅ YAML      |[yamllint](https://megalinter.io/9.5.0/descriptors/yaml_yamllint)                                    |    5|     |     0|       0|       0.46s|


### Notices

📣 **MegaLinter 9.5.0 is out!** Discover the new features and security recommendations in the [release announcement](https://github.com/oxsecurity/megalinter/issues/7835). (Skip this info by defining `SECURITY_SUGGESTIONS: false`)

See detailed reports in MegaLinter artifacts


Your project could benefit from a custom flavor, which would allow you to run only the linters you need, and thus improve runtime performances. (Skip this info by defining `FLAVOR_SUGGESTIONS: false`)

  - Documentation: [Custom Flavors](https://megalinter.io/9.5.0/custom-flavors/)
  - Command: `npx mega-linter-runner@9.5.0 --custom-flavor-setup --custom-flavor-linters PYTHON_RUFF,JSON_V8R,JSON_PRETTIER,MARKDOWN_MARKDOWNLINT,MARKDOWN_MARKDOWN_TABLE_FORMATTER,REPOSITORY_CHECKOV,REPOSITORY_GIT_DIFF,REPOSITORY_GITLEAKS,REPOSITORY_OSV_SCANNER,REPOSITORY_SECRETLINT,REPOSITORY_SYFT,REPOSITORY_TRIVY_SBOM,REPOSITORY_TRUFFLEHOG,YAML_PRETTIER,YAML_YAMLLINT,YAML_V8R`

[![MegaLinter is graciously provided by OX Security](https://raw.githubusercontent.com/oxsecurity/megalinter/main/docs/assets/images/ox-banner.png)](https://www.ox.security/?ref=megalinter)
Show us your support by [**starring ⭐ the repository**](https://github.com/oxsecurity/megalinter)