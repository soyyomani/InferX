include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Helper function to add a tensor test
function(add_tensor_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME} PRIVATE
        InferX::Tensor
        GTest::gtest_main
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Tests — uncommented as components are implemented:
add_tensor_test(test_dtype)
add_tensor_test(test_shape)
add_tensor_test(test_stride)
add_tensor_test(test_storage)
add_tensor_test(test_tensor)
add_tensor_test(test_broadcast)
add_tensor_test(test_iterator)
