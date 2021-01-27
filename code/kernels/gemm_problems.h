// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> training_set = {
    std::make_tuple(1760, 16, 1760, false, false),
    std::make_tuple(1760, 32, 1760, false, false),
    std::make_tuple(1760, 64, 1760, false, false),
    std::make_tuple(1760, 128, 1760, false, false),
    std::make_tuple(1760, 7000, 1760, false, false),
    std::make_tuple(2048, 16, 2048, false, false),
    std::make_tuple(2048, 32, 2048, false, false),
    std::make_tuple(2048, 64, 2048, false, false),
    std::make_tuple(2048, 128, 2048, false, false),
    std::make_tuple(2048, 7000, 2048, false, false),
    std::make_tuple(2560, 16, 2560, false, false),
    std::make_tuple(2560, 32, 2560, false, false),
    std::make_tuple(2560, 64, 2560, false, false),
    std::make_tuple(2560, 128, 2560, false, false),
    std::make_tuple(2560, 7000, 2560, false, false),
    std::make_tuple(4096, 16, 4096, false, false),
    std::make_tuple(4096, 32, 4096, false, false),
    std::make_tuple(4096, 64, 4096, false, false),
    std::make_tuple(4096, 128, 4096, false, false),
    std::make_tuple(4096, 7000, 4096, false, false),
    std::make_tuple(1760, 16, 1760, true, false),
    std::make_tuple(1760, 32, 1760, true, false),
    std::make_tuple(1760, 64, 1760, true, false),
    std::make_tuple(1760, 128, 1760, true, false),
    std::make_tuple(1760, 7000, 1760, true, false),
    std::make_tuple(2048, 16, 2048, true, false),
    std::make_tuple(2048, 32, 2048, true, false),
    std::make_tuple(2048, 64, 2048, true, false),
    std::make_tuple(2048, 128, 2048, true, false),
    std::make_tuple(2048, 7000, 2048, true, false),
    std::make_tuple(2560, 16, 2560, true, false),
    std::make_tuple(2560, 32, 2560, true, false),
    std::make_tuple(2560, 64, 2560, true, false),
    std::make_tuple(2560, 128, 2560, true, false),
    std::make_tuple(2560, 7000, 2560, true, false),
    std::make_tuple(4096, 16, 4096, true, false),
    std::make_tuple(4096, 32, 4096, true, false),
    std::make_tuple(4096, 64, 4096, true, false),
    std::make_tuple(4096, 128, 4096, true, false),
    std::make_tuple(4096, 7000, 4096, true, false),
    std::make_tuple(1760, 7133, 1760, false, true),
    std::make_tuple(2048, 7133, 2048, false, true),
    std::make_tuple(2560, 7133, 2560, false, true),
    std::make_tuple(4096, 7133, 4096, false, true),
    std::make_tuple(5124, 9124, 1760, false, false),
    std::make_tuple(35, 8457, 1760, false, false),
    std::make_tuple(5124, 9124, 2048, false, false),
    std::make_tuple(35, 8457, 2048, false, false),
    std::make_tuple(5124, 9124, 2560, false, false),
    std::make_tuple(35, 8457, 2560, false, false),
    std::make_tuple(5124, 9124, 4096, false, false),
    std::make_tuple(35, 8457, 4096, false, false),
    std::make_tuple(5124, 9124, 1760, true, false),
    std::make_tuple(35, 8457, 1760, true, false),
    std::make_tuple(5124, 9124, 2048, true, false),
    std::make_tuple(35, 8457, 2048, true, false),
    std::make_tuple(5124, 9124, 2560, true, false),
    std::make_tuple(35, 8457, 2560, true, false),
    std::make_tuple(5124, 9124, 4096, true, false),
    std::make_tuple(35, 8457, 4096, true, false),
    std::make_tuple(7680, 16, 2560, false, false),
    std::make_tuple(7680, 32, 2560, false, false),
    std::make_tuple(7680, 64, 2560, false, false),
    std::make_tuple(7680, 128, 2560, false, false),
    std::make_tuple(7680, 16, 2560, true, false),
    std::make_tuple(7680, 32, 2560, true, false),
    std::make_tuple(7680, 64, 2560, true, false),
    std::make_tuple(7680, 128, 2560, true, false),
    std::make_tuple(3072, 16, 1024, false, false),
    std::make_tuple(3072, 32, 1024, false, false),
    std::make_tuple(3072, 64, 1024, false, false),
    std::make_tuple(3072, 128, 1024, false, false),
    std::make_tuple(3072, 16, 1024, true, false),
    std::make_tuple(3072, 32, 1024, true, false),
    std::make_tuple(3072, 64, 1024, true, false),
    std::make_tuple(3072, 128, 1024, true, false),
    std::make_tuple(3072, 7435, 1024, false, true),
    std::make_tuple(7680, 5481, 2560, false, true),
    std::make_tuple(512, 8, 500000, false, false),
    std::make_tuple(1024, 8, 500000, false, false),
    std::make_tuple(512, 16, 500000, false, false),
    std::make_tuple(1024, 16, 500000, false, false),
    std::make_tuple(512, 8, 500000, true, false),
    std::make_tuple(1024, 8, 500000, true, false),
    std::make_tuple(512, 16, 500000, true, false),
    std::make_tuple(1024, 16, 500000, true, false),
    std::make_tuple(1024, 700, 512, false, false),
    std::make_tuple(1024, 700, 512, true, false),
    std::make_tuple(7680, 24000, 2560, false, false),
    std::make_tuple(6144, 24000, 2048, false, false),
    std::make_tuple(4608, 24000, 1536, false, false),
    std::make_tuple(8448, 24000, 2816, false, false),
    std::make_tuple(3072, 24000, 1024, false, false),
    std::make_tuple(7680, 48000, 2560, false, false),
    std::make_tuple(6144, 48000, 2048, false, false),
    std::make_tuple(4608, 48000, 1536, false, false),
    std::make_tuple(8448, 48000, 2816, false, false),
    std::make_tuple(3072, 48000, 1024, false, false),
    std::make_tuple(7680, 24000, 2560, true, false),
    std::make_tuple(6144, 24000, 2048, true, false),
    std::make_tuple(4608, 24000, 1536, true, false),
    std::make_tuple(8448, 24000, 2816, true, false),
    std::make_tuple(3072, 24000, 1024, true, false),
    std::make_tuple(7680, 48000, 2560, true, false),
    std::make_tuple(6144, 48000, 2048, true, false),
    std::make_tuple(4608, 48000, 1536, true, false),
    std::make_tuple(8448, 48000, 2816, true, false),
    std::make_tuple(3072, 48000, 1024, true, false),
    std::make_tuple(6144, 16, 2048, false, false),
    std::make_tuple(4608, 16, 1536, false, false),
    std::make_tuple(8448, 16, 2816, false, false),
    std::make_tuple(6144, 32, 2048, false, false),
    std::make_tuple(4608, 32, 1536, false, false),
    std::make_tuple(8448, 32, 2816, false, false),
    std::make_tuple(6144, 16, 2048, true, false),
    std::make_tuple(4608, 16, 1536, true, false),
    std::make_tuple(8448, 16, 2816, true, false),
    std::make_tuple(6144, 32, 2048, true, false),
    std::make_tuple(4608, 32, 1536, true, false),
    std::make_tuple(8448, 32, 2816, true, false),
    std::make_tuple(512, 24000, 2816, false, false),
    std::make_tuple(512, 24000, 2048, false, false),
    std::make_tuple(512, 24000, 2560, false, false),
    std::make_tuple(512, 24000, 1536, false, false),
    std::make_tuple(1024, 24000, 2816, false, false),
    std::make_tuple(1024, 24000, 2048, false, false),
    std::make_tuple(1024, 24000, 2560, false, false),
    std::make_tuple(1024, 24000, 1536, false, false),
    std::make_tuple(512, 16, 512, false, false),
    std::make_tuple(1024, 16, 512, false, false),
    std::make_tuple(512, 24000, 2816, true, false),
    std::make_tuple(512, 24000, 2048, true, false),
    std::make_tuple(512, 24000, 2560, true, false),
    std::make_tuple(512, 24000, 1536, true, false),
    std::make_tuple(1024, 24000, 2816, true, false),
    std::make_tuple(1024, 24000, 2048, true, false),
    std::make_tuple(1024, 24000, 2560, true, false),
    std::make_tuple(1024, 24000, 1536, true, false),
    std::make_tuple(512, 16, 512, false, true),
    std::make_tuple(1024, 16, 512, false, true),
    std::make_tuple(512, 48000, 2816, false, false),
    std::make_tuple(512, 48000, 2048, false, false),
    std::make_tuple(512, 48000, 2560, false, false),
    std::make_tuple(512, 48000, 1536, false, false),
    std::make_tuple(1024, 48000, 2816, false, false),
    std::make_tuple(1024, 48000, 2048, false, false),
    std::make_tuple(1024, 48000, 2560, false, false),
    std::make_tuple(1024, 48000, 1536, false, false),
    std::make_tuple(512, 32, 512, false, false),
    std::make_tuple(1024, 32, 512, false, false),
    std::make_tuple(512, 48000, 2816, true, false),
    std::make_tuple(512, 48000, 2048, true, false),
    std::make_tuple(512, 48000, 2560, true, false),
    std::make_tuple(512, 48000, 1536, true, false),
    std::make_tuple(1024, 48000, 2816, true, false),
    std::make_tuple(1024, 48000, 2048, true, false),
    std::make_tuple(1024, 48000, 2560, true, false),
    std::make_tuple(1024, 48000, 1536, true, false),
    std::make_tuple(512, 32, 512, false, true),
    std::make_tuple(1024, 32, 512, false, true)
};

// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> inference_server_set = {
    std::make_tuple(5124, 700, 2048, false, false),
    std::make_tuple(35, 700, 2048, false, false),
    std::make_tuple(5124, 700, 2560, false, false),
    std::make_tuple(35, 700, 2560, false, false),
    std::make_tuple(5124, 1500, 2048, false, false),
    std::make_tuple(35, 1500, 2048, false, false),
    std::make_tuple(5124, 1500, 2560, false, false),
    std::make_tuple(35, 1500, 2560, false, false),
    std::make_tuple(7680, 1, 2560, false, false),
    std::make_tuple(7680, 2, 2560, false, false),
    std::make_tuple(7680, 4, 2560, false, false),
    std::make_tuple(3072, 1, 1024, false, false),
    std::make_tuple(3072, 2, 1024, false, false),
    std::make_tuple(3072, 4, 1024, false, false),
    std::make_tuple(512, 1, 500000, false, false),
    std::make_tuple(1024, 1, 500000, false, false),
    std::make_tuple(512, 2, 500000, false, false),
    std::make_tuple(1024, 2, 500000, false, false),
    std::make_tuple(512, 4, 500000, false, false),
    std::make_tuple(1024, 4, 500000, false, false),
    std::make_tuple(1024, 700, 512, false, false),
    std::make_tuple(7680, 1500, 2560, false, false),
    std::make_tuple(6144, 1500, 2048, false, false),
    std::make_tuple(4608, 1500, 1536, false, false),
    std::make_tuple(8448, 1500, 2816, false, false),
    std::make_tuple(3072, 1500, 1024, false, false),
    std::make_tuple(7680, 3000, 2560, false, false),
    std::make_tuple(6144, 3000, 2048, false, false),
    std::make_tuple(4608, 3000, 1536, false, false),
    std::make_tuple(8448, 3000, 2816, false, false),
    std::make_tuple(3072, 3000, 1024, false, false),
    std::make_tuple(7680, 6000, 2560, false, false),
    std::make_tuple(6144, 6000, 2048, false, false),
    std::make_tuple(4608, 6000, 1536, false, false),
    std::make_tuple(8448, 6000, 2816, false, false),
    std::make_tuple(3072, 6000, 1024, false, false),
    std::make_tuple(6144, 1, 2048, false, false),
    std::make_tuple(4608, 1, 1536, false, false),
    std::make_tuple(8448, 1, 2816, false, false),
    std::make_tuple(6144, 2, 2048, false, false),
    std::make_tuple(4608, 2, 1536, false, false),
    std::make_tuple(8448, 2, 2816, false, false),
    std::make_tuple(6144, 4, 2048, false, false),
    std::make_tuple(4608, 4, 1536, false, false),
    std::make_tuple(8448, 4, 2816, false, false),
    std::make_tuple(512, 1500, 2816, false, false),
    std::make_tuple(512, 1500, 2048, false, false),
    std::make_tuple(512, 1500, 2560, false, false),
    std::make_tuple(512, 1500, 1536, false, false),
    std::make_tuple(1024, 1500, 2816, false, false),
    std::make_tuple(1024, 1500, 2048, false, false),
    std::make_tuple(1024, 1500, 2560, false, false),
    std::make_tuple(1024, 1500, 1536, false, false),
    std::make_tuple(512, 1, 512, false, false),
    std::make_tuple(1024, 1, 512, false, false),
    std::make_tuple(512, 3000, 2816, false, false),
    std::make_tuple(512, 3000, 2048, false, false),
    std::make_tuple(512, 3000, 2560, false, false),
    std::make_tuple(512, 3000, 1536, false, false),
    std::make_tuple(1024, 3000, 2816, false, false),
    std::make_tuple(1024, 3000, 2048, false, false),
    std::make_tuple(1024, 3000, 2560, false, false),
    std::make_tuple(1024, 3000, 1536, false, false),
    std::make_tuple(512, 2, 512, false, false),
    std::make_tuple(1024, 2, 512, false, false),
    std::make_tuple(512, 6000, 2816, false, false),
    std::make_tuple(512, 6000, 2048, false, false),
    std::make_tuple(512, 6000, 2560, false, false),
    std::make_tuple(512, 6000, 1536, false, false),
    std::make_tuple(1024, 6000, 2816, false, false),
    std::make_tuple(1024, 6000, 2048, false, false),
    std::make_tuple(1024, 6000, 2560, false, false),
    std::make_tuple(1024, 6000, 1536, false, false),
    std::make_tuple(512, 4, 512, false, false),
    std::make_tuple(1024, 4, 512, false, false)
};

// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> inference_device_set = {
    std::make_tuple(5124, 700, 2048, false, true ),
    std::make_tuple(35, 700, 2048, false, true),
    std::make_tuple(3072, 1, 1024, false, true),
    std::make_tuple(64, 1, 1216, false, true),
    std::make_tuple(3072, 1500, 1024, false, true),
    std::make_tuple(128, 1500, 1280, false, true),
    std::make_tuple(3072, 1500, 128, false, true),
    std::make_tuple(128, 1, 1024, false, true),
    std::make_tuple(3072, 1, 128, false, true),
    std::make_tuple(176, 1500, 1408, false, true),
    std::make_tuple(4224, 1500, 176, false, true),
    std::make_tuple(128, 1, 1408, false, true),
    std::make_tuple(4224, 1, 128, false, true)
};

