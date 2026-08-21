// independent per-status asserts (literal expected; not a SUT clone function).
#[cfg(test)]
#[allow(clippy::missing_assert_message, reason = "status in test name")]
mod http_status_matrix {
    use super::{HttpRetryClass, classify_http_status};
    use crate::openai_compat::http_status_error;
    use ovo_types::ErrorCode;

    #[test]
    fn status_100_is_fatal() {
        assert_eq!(classify_http_status(100, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_105_is_fatal() {
        assert_eq!(classify_http_status(105, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_110_is_fatal() {
        assert_eq!(classify_http_status(110, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_115_is_fatal() {
        assert_eq!(classify_http_status(115, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_120_is_fatal() {
        assert_eq!(classify_http_status(120, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_125_is_fatal() {
        assert_eq!(classify_http_status(125, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_130_is_fatal() {
        assert_eq!(classify_http_status(130, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_135_is_fatal() {
        assert_eq!(classify_http_status(135, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_140_is_fatal() {
        assert_eq!(classify_http_status(140, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_145_is_fatal() {
        assert_eq!(classify_http_status(145, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_150_is_fatal() {
        assert_eq!(classify_http_status(150, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_155_is_fatal() {
        assert_eq!(classify_http_status(155, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_160_is_fatal() {
        assert_eq!(classify_http_status(160, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_165_is_fatal() {
        assert_eq!(classify_http_status(165, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_170_is_fatal() {
        assert_eq!(classify_http_status(170, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_175_is_fatal() {
        assert_eq!(classify_http_status(175, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_180_is_fatal() {
        assert_eq!(classify_http_status(180, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_185_is_fatal() {
        assert_eq!(classify_http_status(185, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_190_is_fatal() {
        assert_eq!(classify_http_status(190, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_195_is_fatal() {
        assert_eq!(classify_http_status(195, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_200_is_fatal() {
        assert_eq!(classify_http_status(200, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_205_is_fatal() {
        assert_eq!(classify_http_status(205, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_210_is_fatal() {
        assert_eq!(classify_http_status(210, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_215_is_fatal() {
        assert_eq!(classify_http_status(215, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_220_is_fatal() {
        assert_eq!(classify_http_status(220, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_225_is_fatal() {
        assert_eq!(classify_http_status(225, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_230_is_fatal() {
        assert_eq!(classify_http_status(230, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_235_is_fatal() {
        assert_eq!(classify_http_status(235, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_240_is_fatal() {
        assert_eq!(classify_http_status(240, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_245_is_fatal() {
        assert_eq!(classify_http_status(245, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_250_is_fatal() {
        assert_eq!(classify_http_status(250, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_255_is_fatal() {
        assert_eq!(classify_http_status(255, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_260_is_fatal() {
        assert_eq!(classify_http_status(260, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_265_is_fatal() {
        assert_eq!(classify_http_status(265, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_270_is_fatal() {
        assert_eq!(classify_http_status(270, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_275_is_fatal() {
        assert_eq!(classify_http_status(275, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_280_is_fatal() {
        assert_eq!(classify_http_status(280, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_285_is_fatal() {
        assert_eq!(classify_http_status(285, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_290_is_fatal() {
        assert_eq!(classify_http_status(290, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_295_is_fatal() {
        assert_eq!(classify_http_status(295, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_300_is_fatal() {
        assert_eq!(classify_http_status(300, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_305_is_fatal() {
        assert_eq!(classify_http_status(305, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_310_is_fatal() {
        assert_eq!(classify_http_status(310, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_315_is_fatal() {
        assert_eq!(classify_http_status(315, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_320_is_fatal() {
        assert_eq!(classify_http_status(320, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_325_is_fatal() {
        assert_eq!(classify_http_status(325, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_330_is_fatal() {
        assert_eq!(classify_http_status(330, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_335_is_fatal() {
        assert_eq!(classify_http_status(335, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_340_is_fatal() {
        assert_eq!(classify_http_status(340, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_345_is_fatal() {
        assert_eq!(classify_http_status(345, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_350_is_fatal() {
        assert_eq!(classify_http_status(350, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_355_is_fatal() {
        assert_eq!(classify_http_status(355, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_360_is_fatal() {
        assert_eq!(classify_http_status(360, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_365_is_fatal() {
        assert_eq!(classify_http_status(365, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_370_is_fatal() {
        assert_eq!(classify_http_status(370, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_375_is_fatal() {
        assert_eq!(classify_http_status(375, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_380_is_fatal() {
        assert_eq!(classify_http_status(380, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_385_is_fatal() {
        assert_eq!(classify_http_status(385, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_390_is_fatal() {
        assert_eq!(classify_http_status(390, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_395_is_fatal() {
        assert_eq!(classify_http_status(395, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_400_is_fatal() {
        assert_eq!(classify_http_status(400, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_401_is_fatal() {
        assert_eq!(classify_http_status(401, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_402_is_fatal() {
        assert_eq!(classify_http_status(402, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_403_is_fatal() {
        assert_eq!(classify_http_status(403, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_404_is_fatal() {
        assert_eq!(classify_http_status(404, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_405_is_fatal() {
        assert_eq!(classify_http_status(405, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_406_is_fatal() {
        assert_eq!(classify_http_status(406, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_407_is_fatal() {
        assert_eq!(classify_http_status(407, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_408_is_fatal() {
        assert_eq!(classify_http_status(408, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_409_is_fatal() {
        assert_eq!(classify_http_status(409, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_410_is_fatal() {
        assert_eq!(classify_http_status(410, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_411_is_fatal() {
        assert_eq!(classify_http_status(411, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_412_is_fatal() {
        assert_eq!(classify_http_status(412, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_413_is_fatal() {
        assert_eq!(classify_http_status(413, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_414_is_fatal() {
        assert_eq!(classify_http_status(414, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_415_is_fatal() {
        assert_eq!(classify_http_status(415, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_416_is_fatal() {
        assert_eq!(classify_http_status(416, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_417_is_fatal() {
        assert_eq!(classify_http_status(417, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_418_is_fatal() {
        assert_eq!(classify_http_status(418, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_419_is_fatal() {
        assert_eq!(classify_http_status(419, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_420_is_fatal() {
        assert_eq!(classify_http_status(420, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_421_is_fatal() {
        assert_eq!(classify_http_status(421, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_422_is_fatal() {
        assert_eq!(classify_http_status(422, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_423_is_fatal() {
        assert_eq!(classify_http_status(423, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_424_is_fatal() {
        assert_eq!(classify_http_status(424, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_425_is_fatal() {
        assert_eq!(classify_http_status(425, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_426_is_fatal() {
        assert_eq!(classify_http_status(426, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_427_is_fatal() {
        assert_eq!(classify_http_status(427, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_428_is_fatal() {
        assert_eq!(classify_http_status(428, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_429_is_ratelimited() {
        assert_eq!(classify_http_status(429, None), HttpRetryClass::RateLimited);
    }

    #[test]
    fn status_430_is_fatal() {
        assert_eq!(classify_http_status(430, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_431_is_fatal() {
        assert_eq!(classify_http_status(431, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_432_is_fatal() {
        assert_eq!(classify_http_status(432, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_433_is_fatal() {
        assert_eq!(classify_http_status(433, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_434_is_fatal() {
        assert_eq!(classify_http_status(434, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_435_is_fatal() {
        assert_eq!(classify_http_status(435, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_436_is_fatal() {
        assert_eq!(classify_http_status(436, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_437_is_fatal() {
        assert_eq!(classify_http_status(437, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_438_is_fatal() {
        assert_eq!(classify_http_status(438, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_439_is_fatal() {
        assert_eq!(classify_http_status(439, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_440_is_fatal() {
        assert_eq!(classify_http_status(440, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_441_is_fatal() {
        assert_eq!(classify_http_status(441, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_442_is_fatal() {
        assert_eq!(classify_http_status(442, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_443_is_fatal() {
        assert_eq!(classify_http_status(443, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_444_is_fatal() {
        assert_eq!(classify_http_status(444, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_445_is_fatal() {
        assert_eq!(classify_http_status(445, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_446_is_fatal() {
        assert_eq!(classify_http_status(446, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_447_is_fatal() {
        assert_eq!(classify_http_status(447, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_448_is_fatal() {
        assert_eq!(classify_http_status(448, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_449_is_fatal() {
        assert_eq!(classify_http_status(449, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_450_is_fatal() {
        assert_eq!(classify_http_status(450, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_451_is_fatal() {
        assert_eq!(classify_http_status(451, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_452_is_fatal() {
        assert_eq!(classify_http_status(452, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_453_is_fatal() {
        assert_eq!(classify_http_status(453, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_454_is_fatal() {
        assert_eq!(classify_http_status(454, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_455_is_fatal() {
        assert_eq!(classify_http_status(455, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_456_is_fatal() {
        assert_eq!(classify_http_status(456, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_457_is_fatal() {
        assert_eq!(classify_http_status(457, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_458_is_fatal() {
        assert_eq!(classify_http_status(458, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_459_is_fatal() {
        assert_eq!(classify_http_status(459, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_460_is_fatal() {
        assert_eq!(classify_http_status(460, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_461_is_fatal() {
        assert_eq!(classify_http_status(461, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_462_is_fatal() {
        assert_eq!(classify_http_status(462, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_463_is_fatal() {
        assert_eq!(classify_http_status(463, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_464_is_fatal() {
        assert_eq!(classify_http_status(464, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_465_is_fatal() {
        assert_eq!(classify_http_status(465, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_466_is_fatal() {
        assert_eq!(classify_http_status(466, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_467_is_fatal() {
        assert_eq!(classify_http_status(467, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_468_is_fatal() {
        assert_eq!(classify_http_status(468, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_469_is_fatal() {
        assert_eq!(classify_http_status(469, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_470_is_fatal() {
        assert_eq!(classify_http_status(470, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_471_is_fatal() {
        assert_eq!(classify_http_status(471, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_472_is_fatal() {
        assert_eq!(classify_http_status(472, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_473_is_fatal() {
        assert_eq!(classify_http_status(473, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_474_is_fatal() {
        assert_eq!(classify_http_status(474, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_475_is_fatal() {
        assert_eq!(classify_http_status(475, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_476_is_fatal() {
        assert_eq!(classify_http_status(476, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_477_is_fatal() {
        assert_eq!(classify_http_status(477, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_478_is_fatal() {
        assert_eq!(classify_http_status(478, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_479_is_fatal() {
        assert_eq!(classify_http_status(479, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_480_is_fatal() {
        assert_eq!(classify_http_status(480, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_481_is_fatal() {
        assert_eq!(classify_http_status(481, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_482_is_fatal() {
        assert_eq!(classify_http_status(482, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_483_is_fatal() {
        assert_eq!(classify_http_status(483, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_484_is_fatal() {
        assert_eq!(classify_http_status(484, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_485_is_fatal() {
        assert_eq!(classify_http_status(485, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_486_is_fatal() {
        assert_eq!(classify_http_status(486, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_487_is_fatal() {
        assert_eq!(classify_http_status(487, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_488_is_fatal() {
        assert_eq!(classify_http_status(488, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_489_is_fatal() {
        assert_eq!(classify_http_status(489, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_490_is_fatal() {
        assert_eq!(classify_http_status(490, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_491_is_fatal() {
        assert_eq!(classify_http_status(491, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_492_is_fatal() {
        assert_eq!(classify_http_status(492, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_493_is_fatal() {
        assert_eq!(classify_http_status(493, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_494_is_fatal() {
        assert_eq!(classify_http_status(494, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_495_is_fatal() {
        assert_eq!(classify_http_status(495, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_496_is_fatal() {
        assert_eq!(classify_http_status(496, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_497_is_fatal() {
        assert_eq!(classify_http_status(497, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_498_is_fatal() {
        assert_eq!(classify_http_status(498, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_499_is_fatal() {
        assert_eq!(classify_http_status(499, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_500_is_retry() {
        assert_eq!(classify_http_status(500, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_501_is_retry() {
        assert_eq!(classify_http_status(501, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_502_is_retry() {
        assert_eq!(classify_http_status(502, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_503_is_retry() {
        assert_eq!(classify_http_status(503, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_504_is_retry() {
        assert_eq!(classify_http_status(504, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_505_is_retry() {
        assert_eq!(classify_http_status(505, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_506_is_retry() {
        assert_eq!(classify_http_status(506, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_507_is_retry() {
        assert_eq!(classify_http_status(507, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_508_is_retry() {
        assert_eq!(classify_http_status(508, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_509_is_retry() {
        assert_eq!(classify_http_status(509, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_510_is_retry() {
        assert_eq!(classify_http_status(510, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_511_is_retry() {
        assert_eq!(classify_http_status(511, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_512_is_retry() {
        assert_eq!(classify_http_status(512, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_513_is_retry() {
        assert_eq!(classify_http_status(513, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_514_is_retry() {
        assert_eq!(classify_http_status(514, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_515_is_retry() {
        assert_eq!(classify_http_status(515, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_516_is_retry() {
        assert_eq!(classify_http_status(516, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_517_is_retry() {
        assert_eq!(classify_http_status(517, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_518_is_retry() {
        assert_eq!(classify_http_status(518, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_519_is_retry() {
        assert_eq!(classify_http_status(519, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_520_is_retry() {
        assert_eq!(classify_http_status(520, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_521_is_retry() {
        assert_eq!(classify_http_status(521, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_522_is_retry() {
        assert_eq!(classify_http_status(522, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_523_is_retry() {
        assert_eq!(classify_http_status(523, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_524_is_retry() {
        assert_eq!(classify_http_status(524, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_525_is_fatal() {
        assert_eq!(classify_http_status(525, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_526_is_fatal() {
        assert_eq!(classify_http_status(526, None), HttpRetryClass::Fatal);
    }

    #[test]
    fn status_527_is_retry() {
        assert_eq!(classify_http_status(527, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_528_is_retry() {
        assert_eq!(classify_http_status(528, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_529_is_retry() {
        assert_eq!(classify_http_status(529, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_530_is_retry() {
        assert_eq!(classify_http_status(530, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_531_is_retry() {
        assert_eq!(classify_http_status(531, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_532_is_retry() {
        assert_eq!(classify_http_status(532, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_533_is_retry() {
        assert_eq!(classify_http_status(533, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_534_is_retry() {
        assert_eq!(classify_http_status(534, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_535_is_retry() {
        assert_eq!(classify_http_status(535, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_536_is_retry() {
        assert_eq!(classify_http_status(536, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_537_is_retry() {
        assert_eq!(classify_http_status(537, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_538_is_retry() {
        assert_eq!(classify_http_status(538, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_539_is_retry() {
        assert_eq!(classify_http_status(539, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_540_is_retry() {
        assert_eq!(classify_http_status(540, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_541_is_retry() {
        assert_eq!(classify_http_status(541, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_542_is_retry() {
        assert_eq!(classify_http_status(542, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_543_is_retry() {
        assert_eq!(classify_http_status(543, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_544_is_retry() {
        assert_eq!(classify_http_status(544, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_545_is_retry() {
        assert_eq!(classify_http_status(545, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_546_is_retry() {
        assert_eq!(classify_http_status(546, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_547_is_retry() {
        assert_eq!(classify_http_status(547, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_548_is_retry() {
        assert_eq!(classify_http_status(548, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_549_is_retry() {
        assert_eq!(classify_http_status(549, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_550_is_retry() {
        assert_eq!(classify_http_status(550, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_551_is_retry() {
        assert_eq!(classify_http_status(551, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_552_is_retry() {
        assert_eq!(classify_http_status(552, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_553_is_retry() {
        assert_eq!(classify_http_status(553, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_554_is_retry() {
        assert_eq!(classify_http_status(554, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_555_is_retry() {
        assert_eq!(classify_http_status(555, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_556_is_retry() {
        assert_eq!(classify_http_status(556, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_557_is_retry() {
        assert_eq!(classify_http_status(557, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_558_is_retry() {
        assert_eq!(classify_http_status(558, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_559_is_retry() {
        assert_eq!(classify_http_status(559, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_560_is_retry() {
        assert_eq!(classify_http_status(560, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_561_is_retry() {
        assert_eq!(classify_http_status(561, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_562_is_retry() {
        assert_eq!(classify_http_status(562, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_563_is_retry() {
        assert_eq!(classify_http_status(563, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_564_is_retry() {
        assert_eq!(classify_http_status(564, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_565_is_retry() {
        assert_eq!(classify_http_status(565, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_566_is_retry() {
        assert_eq!(classify_http_status(566, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_567_is_retry() {
        assert_eq!(classify_http_status(567, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_568_is_retry() {
        assert_eq!(classify_http_status(568, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_569_is_retry() {
        assert_eq!(classify_http_status(569, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_570_is_retry() {
        assert_eq!(classify_http_status(570, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_571_is_retry() {
        assert_eq!(classify_http_status(571, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_572_is_retry() {
        assert_eq!(classify_http_status(572, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_573_is_retry() {
        assert_eq!(classify_http_status(573, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_574_is_retry() {
        assert_eq!(classify_http_status(574, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_575_is_retry() {
        assert_eq!(classify_http_status(575, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_576_is_retry() {
        assert_eq!(classify_http_status(576, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_577_is_retry() {
        assert_eq!(classify_http_status(577, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_578_is_retry() {
        assert_eq!(classify_http_status(578, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_579_is_retry() {
        assert_eq!(classify_http_status(579, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_580_is_retry() {
        assert_eq!(classify_http_status(580, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_581_is_retry() {
        assert_eq!(classify_http_status(581, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_582_is_retry() {
        assert_eq!(classify_http_status(582, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_583_is_retry() {
        assert_eq!(classify_http_status(583, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_584_is_retry() {
        assert_eq!(classify_http_status(584, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_585_is_retry() {
        assert_eq!(classify_http_status(585, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_586_is_retry() {
        assert_eq!(classify_http_status(586, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_587_is_retry() {
        assert_eq!(classify_http_status(587, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_588_is_retry() {
        assert_eq!(classify_http_status(588, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_589_is_retry() {
        assert_eq!(classify_http_status(589, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_590_is_retry() {
        assert_eq!(classify_http_status(590, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_591_is_retry() {
        assert_eq!(classify_http_status(591, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_592_is_retry() {
        assert_eq!(classify_http_status(592, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_593_is_retry() {
        assert_eq!(classify_http_status(593, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_594_is_retry() {
        assert_eq!(classify_http_status(594, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_595_is_retry() {
        assert_eq!(classify_http_status(595, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_596_is_retry() {
        assert_eq!(classify_http_status(596, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_597_is_retry() {
        assert_eq!(classify_http_status(597, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_598_is_retry() {
        assert_eq!(classify_http_status(598, None), HttpRetryClass::Retry);
    }

    #[test]
    fn status_599_is_retry() {
        assert_eq!(classify_http_status(599, None), HttpRetryClass::Retry);
    }

    #[test]
    fn live_path_429_is_rate_limit_code() {
        assert_eq!(http_status_error(429, "x").code(), ErrorCode::LlmRateLimit);
    }
    #[test]
    fn live_path_401_is_auth() {
        assert_eq!(http_status_error(401, "x").code(), ErrorCode::LlmAuth);
    }
    #[test]
    fn live_path_500_retries() {
        assert_eq!(
            http_status_error(500, "x").retry_class(),
            ovo_types::RetryClass::Backoff
        );
    }
    #[test]
    fn live_path_525_fatal() {
        assert_eq!(
            http_status_error(525, "x").retry_class(),
            ovo_types::RetryClass::Never
        );
    }
    #[test]
    fn x_should_retry_true_418() {
        assert_eq!(classify_http_status(418, Some(true)), HttpRetryClass::Retry);
    }
    #[test]
    fn x_should_retry_false_500() {
        assert_eq!(classify_http_status(500, Some(false)), HttpRetryClass::Fatal);
    }
}
