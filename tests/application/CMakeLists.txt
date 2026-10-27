include(cc_test)

cc_test(
  NAME
    chat_template_renderer_test
  SRCS
    chat_template_renderer_test.cpp
  DEPS
    :chat_template_renderer
    GTest::gtest_main
)
