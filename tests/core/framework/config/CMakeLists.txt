include(cc_test)

cc_test(
  NAME
    config_json_test
  SRCS
    config_json_test.cpp
  DATA
    config_test.json
  DEPS
    :config
    :util
    GTest::gtest_main
)

cc_test(
  NAME
    disagg_pd_config_test
  SRCS
    disagg_pd_config_test.cpp
  DEPS
    :config
    GTest::gtest_main
    glog::glog
)

cc_test(
  NAME
    rec_execution_config_test
  SRCS
    rec_execution_config_test.cpp
  DEPS
    GTest::gtest_main
)
