module AriadneSparseConnectivityTracerExt

using Ariadne
import SparseConnectivityTracer

function Ariadne.jacobian_sparsity(
        f!, u, p; detector = SparseConnectivityTracer.TracerSparsityDetector(),
        res = similar(u)
    )
    return SparseConnectivityTracer.jacobian_sparsity((y, x) -> f!(y, x, p), res, u, detector)
end

end # module
