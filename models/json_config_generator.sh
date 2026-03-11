model_name="${1}"
training_samples="${2}"
test_samples="${3}"
features="${4}"
output_dir="${5}"
rand="${6}"
model_specific="${7}"
task="${8}"
analysis_name="${9}"
json_config_file="${10}"


echo "{ " > "${json_config_file}"

echo "\"model_name\": \"${model_name}\"," >> "${json_config_file}"
echo "\"training_samples\": \"${training_samples}\"," >> "${json_config_file}"
echo "\"test_samples\": \"${test_samples}\"," >> "${json_config_file}"
echo "\"features\": \"${features}\"," >> "${json_config_file}"
echo "\"task\": \"${task}\"," >> "${json_config_file}"
echo "\"analysis_name\": \"${analysis_name}\"," >> "${json_config_file}"
echo "\"output_dir\": \"${output_dir}\"," >> "${json_config_file}"
echo "\"rand\": \"${rand}\"," >> "${json_config_file}"
echo "\"model_specific\": ${model_specific}" >> "${json_config_file}"


echo " }" >> "${json_config_file}"


