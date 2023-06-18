#include "../../shared.h"
#include "cuknl_shared.h"

void check_errors(int line_num, const char* file)
{
    hipDeviceSynchronize();

    int result = hipGetLastError();

    if(result != hipSuccess)
    {
        die(line_num, file, "Error in %s - return code %d (%s)\n",
                file, result, cuda_codes(result));
    }
}

// Enumeration for the set of potential CUDA error codes.
const char* cuda_codes(int code)
{
	switch(code)
	{
		case hipSuccess: return "hipSuccess"; // 0
		case hipErrorMissingConfiguration: return "hipErrorMissingConfiguration"; // 1
		case hipErrorOutOfMemory: return "hipErrorOutOfMemory"; // 2
		case hipErrorNotInitialized: return "hipErrorNotInitialized"; // 3
		case hipErrorLaunchFailure: return "hipErrorLaunchFailure"; // 4
		case hipErrorPriorLaunchFailure: return "hipErrorPriorLaunchFailure"; // 5
		case hipErrorLaunchTimeOut: return "hipErrorLaunchTimeOut"; // 6
		case hipErrorLaunchOutOfResources: return "hipErrorLaunchOutOfResources"; // 7
		case hipErrorInvalidDeviceFunction: return "hipErrorInvalidDeviceFunction"; // 8
		case hipErrorInvalidConfiguration: return "hipErrorInvalidConfiguration"; // 9
		case hipErrorInvalidDevice: return "hipErrorInvalidDevice"; // 10
		case hipErrorInvalidValue: return "hipErrorInvalidValue";// 11
		case hipErrorInvalidPitchValue: return "hipErrorInvalidPitchValue";// 12
		case hipErrorInvalidSymbol: return "hipErrorInvalidSymbol";// 13
		case hipErrorMapFailed: return "hipErrorMapFailed";// 14
		case hipErrorUnmapFailed: return "hipErrorUnmapFailed";// 15
//		case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";// 16
		case hipErrorInvalidDevicePointer: return "hipErrorInvalidDevicePointer";// 17
//		case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";// 18
//		case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";// 19
//		case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";// 20
		case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";// 21
//		case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";// 22
//		case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";// 23
//		case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";// 24
//		case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";// 25
//		case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";// 26
//		case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";// 27
//		case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";// 28
		case hipErrorDeinitialized: return "hipErrorDeinitialized";// 29
		case hipErrorUnknown: return "hipErrorUnknown";// 30
//		case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";// 31
//		case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";// 32
		case hipErrorInvalidHandle: return "hipErrorInvalidHandle";// 33
		case hipErrorNotReady: return "hipErrorNotReady";// 34
		case hipErrorInsufficientDriver: return "hipErrorInsufficientDriver";// 35
		case hipErrorSetOnActiveProcess: return "hipErrorSetOnActiveProcess";// 36
//		case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";// 37
		case hipErrorNoDevice: return "hipErrorNoDevice";// 38
		case hipErrorECCNotCorrectable: return "hipErrorECCNotCorrectable";// 39
		case hipErrorSharedObjectSymbolNotFound: return "hipErrorSharedObjectSymbolNotFound";// 40
		case hipErrorSharedObjectInitFailed: return "hipErrorSharedObjectInitFailed";// 41
		case hipErrorUnsupportedLimit: return "hipErrorUnsupportedLimit";// 42
//		case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";// 43
//		case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";// 44
//		case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";// 45
//		case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";// 46
		case hipErrorInvalidImage: return "hipErrorInvalidImage";// 47
		case hipErrorNoBinaryForGpu: return "hipErrorNoBinaryForGpu";// 48
//		case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";// 49
		case hipErrorPeerAccessAlreadyEnabled: return "hipErrorPeerAccessAlreadyEnabled";// 50
		case hipErrorPeerAccessNotEnabled: return "hipErrorPeerAccessNotEnabled";// 51
		case hipErrorContextAlreadyInUse: return "hipErrorContextAlreadyInUse";// 52
		case hipErrorProfilerDisabled: return "hipErrorProfilerDisabled";// 53
		case hipErrorProfilerNotInitialized: return "hipErrorProfilerNotInitialized";// 54
		case hipErrorProfilerAlreadyStarted: return "hipErrorProfilerAlreadyStarted";// 55
		case hipErrorProfilerAlreadyStopped: return "hipErrorProfilerAlreadyStopped";// 56
		case hipErrorAssert: return "hipErrorAssert";// 57
//		case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers";// 58
		case hipErrorHostMemoryAlreadyRegistered: return "hipErrorHostMemoryAlreadyRegistered";// 59
		case hipErrorHostMemoryNotRegistered: return "hipErrorHostMemoryNotRegistered";// 60
		case hipErrorOperatingSystem: return "hipErrorOperatingSystem";// 61
//		case cudaErrorStartupFailure: return "cudaErrorStartupFailure";// 62
//		case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";// 63
		default: return "Unknown error";
	}
}


