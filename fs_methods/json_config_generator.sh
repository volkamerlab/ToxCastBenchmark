fs_name="${1}"
num_features="${2}"
features="${3}"
output_dir="${4}"
samples="${5}"
model_specific="${6}"
feature_type="${7}"
json_config_file="${8}"

echo "{ " > "${json_config_file}"

echo "\"fs_name\": \"${fs_name}\"," >> "${json_config_file}"
echo "\"num_features\": \"${num_features}\"," >> "${json_config_file}"
echo "\"features\": \"${features}\"," >> "${json_config_file}"
echo "\"output_dir\": \"${output_dir}\"," >> "${json_config_file}"
echo "\"samples\": \"${samples}\"," >> "${json_config_file}"
echo "\"feature_type\": \"${feature_type}\"," >> "${json_config_file}"
echo "\"fs_specific\": ${model_specific}" >> "${json_config_file}"

echo " }" >> "${json_config_file}"


