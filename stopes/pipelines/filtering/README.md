# NLLB Bitext Filtering

*NB*: This is legacy code that is older than the rest of `stopes`. It has not been
ported yet -- do not depend on it as it will eventually be refactored.

The `filter.py` pipeline applies various filters to bitext, with optional support for
monolingual text. It is configured with [Hydra](https://hydra.cc/).

Before being able to run the filtering pipeline, you will need to create configuration
files containing the list of corpora and the set of length factors. This can be done
by running `scripts/populate_data_conf.py` and `scripts/compute_length_factors.py`,
respectively. Please consult the help of those scripts (by running them with `-h`) to
learn more about how to configure them.

A basic run using default parameters might look like this:
```
python filter.py \
  output_dir=/home/$USER/filter_test \
  data_conf_dir=/home/$USER/data_conf
```
This command will run using the output directory and `data_conf_dir` directory (the
location where the `populate_data_conf.py` and `compute_length_factors.py` scripts
output their configuration files) as specified above, and will additionally load the
default example config `conf/example.yaml`. Anything not specified on the command line
or in `conf/example.yaml` will be set to the default values specified in
`data_types.FilterConfig`.

When needing to run a new filtering job with many parameter overrides, instead of
manually overriding parameters on the command line it is better to create an entirely
new config file, e.g. `conf/my_config.yaml`, containing all overrides. The script can
then be instructed to load it as follows:
```
python filter.py \
  --config-name=my_config \
  output_dir=/home/$USER/filter_test \
  data_conf_dir=/home/$USER/data_conf
```
