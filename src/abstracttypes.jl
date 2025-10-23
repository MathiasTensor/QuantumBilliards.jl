
abstract type CoordinateSystem end
abstract type AbsBasis end
abstract type AbsSolver end
abstract type AbsPoints end
abstract type SweepSolver <: AbsSolver end
abstract type AcceleratedSolver <: AbsSolver end

abstract type AbsState end
abstract type StationaryState <: AbsState end
#grid iterators
abstract type AbsGrid end
