module profile {
    use Map;
    use Time;

    config param enableProfiling = false;
    
    class ProfileTracker {
        var timers: domain(string);
        var timerVals: [timers] stopwatch;
        var callCounts: [timers] int;

        proc startTimer(name: string) {
            if !timers.contains(name) { 
                timers += name;
                timerVals[name] = new stopwatch();
                callCounts[name] = 0;
            }
            timerVals[name].start();
            callCounts[name] += 1;
        }

        proc stopTimer(name: string) {
            if timers.contains(name) {
                timerVals[name].stop();
            } else {
                writeln("Warning: Attempted to stop a timer that was never started: ", name);
            }
        }
    }

    // Global tracker
    private var profiler: owned ProfileTracker?;

    proc initProfiling() {
        if enableProfiling then
            profiler = new ProfileTracker();
    }

    proc startProfiling(name: string) {
        if enableProfiling then
            profiler!.startTimer(name);
    }

    proc stopProfiling(name: string) {
        if enableProfiling then
            profiler!.stopTimer(name);
    }

    proc reportProfiling() {
        if enableProfiling {
            writeln(" -------------------------------------------------------------");
            writeln();
            writeln(" Profiling Results:");
            writeln();
            writef(" %<30s%8s%20s\n", "Kernel Name", "Calls", "Runtime (s)");
            var totalElapsedTime = 0.0;
            for name in profiler!.timers {
                const elapsed = profiler!.timerVals[name].elapsed();
                writef(" %<30s%>8i%>20.3dr\n", name, profiler!.callCounts[name], elapsed);
                totalElapsedTime += elapsed;
            }
            writeln();
            writef(" Total elapsed time: %.3drs.\n", totalElapsedTime);
            writeln();
            writeln(" -------------------------------------------------------------");
        }
    }
}