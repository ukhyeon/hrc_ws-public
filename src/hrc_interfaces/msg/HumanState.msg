# Header에는 시간(stamp)과 좌표계(frame_id) 정보가 포함됨
std_msgs/Header header

# SMPL 추론 결과 (필요하면 추가, 일단은 비워둠)
float32[] effmass

float32[] generalized_massmatrix
float32[] analytic_jp
float32[] analytic_jr
float32[] contact_index
float32[] constarind_index

