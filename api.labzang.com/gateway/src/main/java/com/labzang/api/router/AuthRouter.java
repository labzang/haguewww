package com.labzang.api.router;

import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.reactive.CorsWebFilter;
import org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource;

import java.util.Arrays;
import java.util.Collections;

@Configuration
public class AuthRouter {

    // 라우팅은 application.yaml에서 관리하므로 주석 처리
    // @Bean
    // public RouteLocator authRoutes(RouteLocatorBuilder builder) {
    // return builder.routes()
    // .route("auth-service", r -> r
    // .path("/api/auth/**")
    // .filters(f -> f.rewritePath("/api/(?<segment>.*)", "/${segment}"))
    // .uri("http://authservice:8081"))
    // .build();
    // }

    @Bean
    public CorsWebFilter corsWebFilter() {
        CorsConfiguration corsConfig = new CorsConfiguration();
        // Spring Boot 3.x에서는 setAllowedOriginPatterns 권장 (allowCredentials와 함께 사용 가능)
        corsConfig.setAllowedOriginPatterns(Collections.singletonList("http://localhost:3000"));
        corsConfig.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"));
        corsConfig.setAllowedHeaders(Arrays.asList("*"));
        corsConfig.setExposedHeaders(Arrays.asList("*"));
        corsConfig.setAllowCredentials(true);
        corsConfig.setMaxAge(3600L);

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", corsConfig);

        return new CorsWebFilter(source);
    }
}
